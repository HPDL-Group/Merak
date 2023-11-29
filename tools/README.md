# Introduction

This folder is a collection of scripts for converting checkpoints of one training framework (e.g., DeepSpeed/Megatron/Transformers) into that of a different framework (e.g., Megatron-LM, HF Transformers, Merak).

The folder also contains scripts for inspecting checkpoint files and folders, which could be useful when developing checkpoint conversion logic. At the time of creation, this folder contains scripts to convert DeepSpeed/Megatron/Transformers checkpoints to Merak and HF Transformers checkpoints (this motivated this effort as part of the BigScience project).

The following is the details of checkpoint conversions provided by the available scripts.

## Megatron-DeepSpeed/transformers/Merak to Merak

Alternatively, you can convert first from Megatron-DeepSpeed to Merak:

```bash
# 1. Megatron-DeepSpeed to Merak
python deepspeed_to_transformers.py --merak_tp 2 --merak_pp 4 \
--input_folder /path/to/Megatron-Deepspeed/checkpoint/global_step97500 \
--output_folder /path/to/checkpoints/ds_to_merak 

# 2. Merak to Merak
python convert_merak_gpt2_checkpoint.py --merak_tp 4 --merak_pp 4 \
--load_path /path/to/ds_to_merak/release \
--save_path /path/to/merak_to_merak \
--convert_type merak

# 2.1. Merak to Merak by custom partition
python convert_merak_gpt2_checkpoint.py --merak_tp 4 --merak_pp 4 \
--load_path /path/to/ds_to_merak/release \
--save_path /path/to/merak_to_merak \
--partition_method custom \
--custom_partition 0,3,7,11,15 \
--convert_type merak

# 3. Merak to Transformers
python convert_merak_gpt2_checkpoint.py \
--load_path /path/to/merak_to_merak/release \
--save_path /path/to/merak_to_transformers \
--convert_type transformers

# 4. Transformers to Merak
python convert_merak_gpt2_checkpoint.py --merak_tp 4 --merak_pp 4 \
--load_path /path/to/merak_to_transformers/pytorch_model.bin \
--json_path /path/to/merak_to_transformers/config.json \
--save_path /path/to/transformers_to_merak \
--convert_type transformers_to_merak

```

## Optional arguments introduction
```bash

optional arguments:
  -h, --help            show this help message and exit
  --input_folder INPUT_FOLDER
                        Input DeepSpeed Checkpoint folder
  --output_folder OUTPUT_FOLDER
                        Output Megatron checkpoint folder
  --for_release         Convert for release purpose, reset some (progress)
                        counters.

  --load_path 
                        Load Megatron/transformers/Merak checkpoint folder
  --save_path
                        Save the path for transformers/Merak checkpoints
  --convert_type
                        Typr of convert: {"merak": merak to merak, "transformers": merak to transformers, "transformers_to_merak", transformers to merak}
  --merak_tp TARGET_TP
                        Target TP degree
  --merak_pp TARGET_PP
                        Target PP degree

```