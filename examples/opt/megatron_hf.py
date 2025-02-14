# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
# https://github.com/NVIDIA/Megatron-LM/blob/52e6368/pretrain_gpt.py
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pretrain OPT"""
import os

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import ModelType
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.model.gpt_model import post_language_model_processing


def get_model(
    model_name,
    padded_vocab_size=None,
    disable_flash_attn=False,
    fp16=True,
    ckpt_ratio=0.0,
    impl="slapo",
    delay_init=True,
):
    from transformers import AutoConfig, OPTModel

    config = AutoConfig.from_pretrained(model_name)
    if padded_vocab_size is not None:
        config.vocab_size = padded_vocab_size
    config.use_cache = False

    if "slapo" in impl:
        import slapo
        from slapo.utils.report import report_memory
        from model import schedule_model

        report_memory()
        with slapo.init_empty_weights(enable=delay_init):
            model = OPTModel(config)
        report_memory()
        print(model)
        sch = schedule_model(
            model,
            config,
            disable_flash_attn=disable_flash_attn,
            fp16=fp16,
            ckpt_ratio=ckpt_ratio,
            delay_init=delay_init,
        )
        model, _ = slapo.build(sch)
        print(model)
        report_memory()

    elif impl == "torchscript":
        if ckpt_ratio > 0:
            raise RuntimeError("TorchScript cannot support ckpt")

        config.torchscript = True
        model = OPTModel(config=config)
        if fp16:
            model.half()
        model.cuda()

        bs = 8
        seq_length = 512
        device = "cuda"
        input_ids = torch.ones(bs, seq_length, dtype=torch.long, device=device)
        attention_mask = torch.ones(bs, seq_length, dtype=torch.long, device=device)
        model = torch.jit.trace(model, [input_ids, attention_mask])

    elif impl == "eager":
        model = OPTModel(config)
        if ckpt_ratio > 0:
            model.gradient_checkpointing_enable()

    else:
        raise RuntimeError(f"Unrecognized impl `{impl}`")

    if fp16:
        model.half()
    model.cuda()
    return model


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()
    model_name = os.environ.get("MODEL_NAME", None)
    if model_name is None:
        raise RuntimeError(f"'MODEL_NAME' not found in environment")
    if "opt" not in model_name:
        raise RuntimeError(f"Only opt is supported for now, got {model_name}")
    disable_flash_attn = bool(int(os.environ.get("DISABLE_FLASH_ATTN", "0")))
    ckpt_ratio = 0.0
    if args.recompute_granularity is not None:
        ckpt_ratio = os.environ.get("ckpt_ratio", 1.0)
        if ckpt_ratio == "selective":
            raise NotImplementedError
        ckpt_ratio = 1.0 if ckpt_ratio == "full" else float(ckpt_ratio)

    impl = os.environ.get("IMPL", None)
    if impl is None:
        raise RuntimeError("'IMPL' not found in environment")

    class OPTWithLMHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.opt = get_model(
                model_name,
                args.padded_vocab_size,
                disable_flash_attn,
                args.fp16,
                ckpt_ratio,
                impl,
            )

        def set_input_tensor(self, input_tensor):
            # We don't support Megatron pipeline so this has no effect.
            pass

        def forward(
            self,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
        ):
            assert token_type_ids is None, "Not traced"
            if isinstance(self.opt, torch.jit.ScriptModule):
                output = self.opt(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            else:
                output = self.opt(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
            if isinstance(output, dict):
                lm_output = output["last_hidden_state"].transpose(0, 1).contiguous()
            elif isinstance(output, tuple):
                lm_output = output[0].transpose(0, 1).contiguous()
            else:
                raise RuntimeError

            output_tensor = post_language_model_processing(
                lm_output,
                labels,
                self.opt.decoder.embed_tokens.weight,
                True,
                args.fp16_lm_cross_entropy,
            )
            return output_tensor

    model = OPTWithLMHead()
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["text"]
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["text"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    batch_size = tokens.shape[0]
    seq_length = tokens.shape[1]
    # The shape of attention_mask is (1, 1, seq_length, seq_length) while the 3rd dim
    # is broadcast. The required shape for HF OPT is (batch, seq_length).
    # (1, 1, S, S) -> (1, 1, S, 1)
    attention_mask = attention_mask[..., -1:]
    # (1, 1, S, 1) -> (1, 1, 1, S)
    attention_mask = attention_mask.reshape((1, 1, 1, seq_length))
    # (1, 1, 1, S) -> (B, 1, 1, S)
    attention_mask = attention_mask.repeat(batch_size, 1, 1, 1)
    # (B, 1, 1, S) -> (B, S)
    attention_mask = attention_mask.squeeze(dim=1)
    attention_mask = attention_mask.squeeze(dim=1)

    return tokens, labels, loss_mask, attention_mask, position_ids


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets " "for OPT ...")
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
    )
    print_rank_0("> finished creating OPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
