# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Bert with model schedule."""
import inspect

import slapo
from slapo.logger import get_logger
from schedule import (
    broadcast_input,
    checkpoint,
    replace_and_shard_attention,
    shard_mlp,
    shard_word_embedding,
)

logger = get_logger("Bert")

def schedule_model(
    model,
    config,
    prefix="",
    disable_flash_attn=False,
    fp16=True,
    ckpt_ratio=0.0,
    group=None,
    bcast_input=False,
    pipeline_cuts=None,
    delay_init=True,
):
    if fp16:
        logger.info("Change model dtype to fp16", ranks=0)
        model.half()

    sch = slapo.create_schedule(model, group=group)
    logger.info(f"Scheduling Bert with TP={sch.world_size}", ranks=0)

    # Replace self attention with flash attention, and shard QKV/output
    # if MP group > 1.
    if not disable_flash_attn:
        cnt = replace_and_shard_attention(sch[prefix], config, delay_init=delay_init)
        logger.info(f"Replace {cnt} attention patterns", ranks=0)
    else:
        raise NotImplementedError("Not implemented yet")

    # Shard other parameters if MP group > 1.
    if sch.world_size > 1:
        shard_mlp(sch[prefix], config)
        shard_word_embedding(sch[prefix], config.vocab_size)

        # Broadcast input to all devices within the MP group.
        # This is not required when running on Megatron.
        if bcast_input:
            broadcast_input(sch)

    # Insert activation checkpoints.
    if ckpt_ratio > 0.0:
        n_ckpt = checkpoint(sch[prefix], config, ckpt_ratio=ckpt_ratio)
        logger.info(f"Checkpointing {n_ckpt} layers", ranks=0)

    # Cut pipeline stages.
    if pipeline_cuts:
        input_names = [
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ]
        sig = inspect.signature(model.forward)
        concrete_args = {
            p.name: p.default
            for p in sig.parameters.values()
            if p.name not in input_names
        }
        _prefix = f"{prefix}." if prefix else ""
        sch.trace_for_pipeline(
            f"{_prefix}encoder", tracer="huggingface", concrete_args=concrete_args
        )
        for cut in pipeline_cuts:
            sch[f"{_prefix}encoder.layer.{cut}"].cut_pipeline_stage()

    return sch
