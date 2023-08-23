#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# This script is used to run GPT2 training with 3D parallelism enabled.
# It is tested on a single AWS p3d instance with 8*V100 GPUs.
CUDA_LAUNCH_BLOCKING=0 deepspeed deepspeed_hf.py \
	--batch_size 32 \
	--micro_batch_size 4 \
	--model_name gpt2-xl \
	--iter_nums 20 \
	--hidden-size 7168 \
	--nlayers 30 \
	--num-attn-heads 56 \
	--dropout 0.1 \
	--activation_function gelu \
	--seq_len 2048 \
	--disable_pipeline \
	--checkpoint_method uniform \
	--checkpoint 1.0 \
	--bf16 \
	--attn_op_name cuda \
	--disable_schedule \
	--tmp 1 \
	--pmp 1
