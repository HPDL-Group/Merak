# Zero Consolidate Checkpoint example

This folder contains example scripts that convert multiple sharded zero-1 optimizer checkpoints into one consolidated checkpoint, and use consolidated checkpoint for training with different DP degree from original checkpoint.



## Convert to consolidate checkpoint

```bash
cd ~/merak/tools
bash merge.sh
```

Set the path to the zero checkpoints and the consolidated checkpoint before executing the scripts. Need to specify `.pt` files.

```bash
python consolidate_state_dict.py \
  --checkpoints "/path/to/zero_checkpoints"
  --output  "/path/to/consolidated_checkpoint"
```

## Enable zero re-shard for new DP degree

Add argument for zero re-shard.

```bash
torchrun --nproc_per_node=4 run_bert.py \
         --model-name bert-large \
         --output_dir ./output \
         --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
         --logging_steps 10 \
         --input_name input_ids attention_mask token_type_ids \
         --dataloader_num_workers 2
         --resume_from_checkpoint "/path/to/model_checkpint/"
         --zero_reshard "/path/to/consolidated_checkpoint"
```