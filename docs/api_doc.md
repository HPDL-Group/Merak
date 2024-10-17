<!---
Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.

Maintainer: TXacs (txacs1993@gmail.com), Swli (lucasleesw9@gmail.com), Yck(eyichenke@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
## Merak API

### *Merak.init*

> Initialized the distributed communication groups, include data parallel, model parallel and pipeline parallel. Each parallel degree has it own communication group, we can ge the rank or size through mpu API.

Parameters:

-   dp (int) -- Parallel degree of data parallelism.
-   tp (int) -- Parallel degree of tensor model parallelism.
-   pp (int) -- Parallel degree of pipeline model parallelism.




### class *Merak.MerakArguments*

> Class of Merak's arguments was derived from *TrainingArguments*  of huggingface transformers for convenience. We provide some argument for user, to set method of splitting model.

Extra parameters:

-   train_schedule (str, Optional,  defaults to '1f1b') -- Possible choices are the pipe schedules as strings: '1f1b','pre_recompute_1f1b','ds_default','last_no_recompute_1f1b','shifted_critical_path'.
-   partition_method (str, Optional, defaults to 'uniform') -- Possible choices are the pipeline layer partition strategy as strings: 'uniform','uniform_floor','parameters'.
-   split_method(str, Optional, defaults to 'farthest_min_deps') -- Possible choices are graph partion method as strings: 'farthest_min_deps','layer_split','nearest_min_deps'.
-   custom_split_points(List(str), defaults to None) -- Create split points for layer_split method, default is None.
-   trace_method(str, defaults to 'fx') -- None refers to no tracing, 'fx' refers to use torch.fx for tracing, 'dynamo' refers to use torch._dynamo for tracing.
-   trace_model(str, Optional, defaults to '') -- Add new trace module. example: --trace_model 'Qwen2ForCausalLM'.
-   init_method_std (float, defaults to 0.02) -- Standard deviation of the zero mean normal distribution used for tp weight initialization in Megatron.
-   activation_checkpointing (bool, defaults to True) -- Whether to use activation checkpointing. 
-   checkpoint_num_layers (int, defaults to 1) -- Chunk size (number of layers) for checkpointing.
-   input_names (List[str], Optional, defaults to None) -- The names of the inputs of the traced model. If unset, model.dummy_inputs().keys() are used instead. 
                                                           Example: ['input_ids', 'attention_mask', 'token_type_ids']
-   num_layers (int, Optional, defaults to None) -- Number of hidden layers in the Transformer, will try to get this in model config.
-   seq_length (int, Optional, defaults to None) -- The maximum sequence length that this model might ever be used with, will try to get this in model config.
-   num_heads (int, Optional, defaults to None) -- The number of heads that this model might ever be used with, will try to get this in model config. Defaults to None.
-   wall_clock_breakdown (bool, defaults to False) -- Whether to log detail time spend on each rank.
-   shard_count (int, Optional, defaults to None) -- Number of shards that model needs to be break, will be training_args.num_layers*2 if not set.
-   prescale_gradients (bool, defaults to False) -- Whether to enable gradient prescaling.
-   gradient_predivide_factor (float, defaults to 1.0) -- Gradient predivide factor in gradient prescaling.
-   zero_allow_untested_optimizer (bool, defaults to False) -- Whether to allow wrap untested optimizer. The untested optimizer does not guarantee the correctness of training.
-   zero_stage (float, defaults to 1) -- Stage of zero optimization.
-   zero_allgather_bucket_size (float, defaults to 500000000) -- The bucket size per communication in optimzier step.
-   zero_reduce_bucket_size (float, defaults to 500000000) -- The bucket size per communication in gradients reduce.
-   save (bool, defaults to False) -- Whether to save checkpoint.
-   finetune (bool, defaults to False) -- Load model for finetuning. Do not load optimizer or rng state from checkpoint and set iteration to 0. Assumed when loading a release checkpoint.
-   no_save_rng (bool, defaults to False) -- Do not save current rng state.
-   no_save_optim (bool, defaults to False) -- Do not save current optimizer.
-   no_load_rng (bool, defaults to False) -- Do not load current optimizer.
-   no_load_optim (bool, defaults to False) -- Do not load current optimizer.
-   split_inputs (bool, defaults to False) -- Whether to split input data.
-   parallel_vocab (bool, defaults to False) -- Whether to parallel vocabulary when TMP > 1.
-   sequence_parallel (bool, defaults to False) -- Whether to use sequence parallel when TMP > 1.
-   sequence_dim (int, defaults to 1) -- Sequence length dimension in hidden states.
-   dealloc_pipeoutput (bool, defaults to False) -- Whether to dealloc pipeline sended activation output.
-   activation_checkpoint_ratio (float, Optional, defaults to None) -- activation checkpoint ratio of first stage, in range(0,1). Default to None.
-   tp_overlapping_level (float, Optional, defaults to 0) -- "Possible tensor parallelism communication overlapping level from 0 to 3.
                                                              0 refers to no overlapping; 1 refers to only overlap within linear function;
                                                              2 refers to overlap within transformer blocks, requires rewrite transformer blocks;
                                                              3 refers to overlap between transformer blocks, requires rewrite transformer model.,
                                                              choices: [0,1,2,3]"
-   loss_scale (float, defaults to 0) -- loss_scale is a fp16 parameter representing the loss scaling value for FP16 training.
                                         The default value of 0.0 results in dynamic loss scaling,
                                         otherwise the value will be used for static fixed loss scaling.
-   initial_scale_power (int, defaults to 32) -- 'initial_scale_power is a fp16 parameter representing the power of the initial dynamic loss scale value.
                                                  The actual loss scale is computed as 2^initial_scale_power.'
-   loss_scale_window (int, defaults to 1000) -- 'loss_scale_window is a fp16 parameter representing the window over which to raise/lower the dynamic loss scale value.'
-   hysteresis (int, defaults to 2) -- 'hysteresis is a fp16 parameter representing the delay shift in dynamic loss scaling.'
-   min_loss_scale (int, defaults to 1) -- 'min_loss_scale is a fp16 parameter representing the minimum dynamic loss scale value.'
-   custom_partition (float, str, defaults to None) -- 'Customize the partition size of the model. Length of list is pipeline_world_size + 1.
                                                       'Example: [0, 6, 12, 18, 26, ..., last_layer_idx]', Default to None.
-   no_tie_modules (bool, defaults to False) -- Whether to set tie modules.
-   save_total_limit (int, defaults to -1) -- Limit the max numbers of checkpoints.
-   eval_iters (int, defaults to None) -- Number of iterations to run for evaluationvalidation/test for.
-   text_generation (bool, Optional, defaults to False) -- Whether to do text generate.
-   out_seq_length (int, Optional, defaults to 1024) -- The maximum sequence length that this model's output. Defaults to 1024.
-   temperature (float, Optional, defaults to 0.9) -- Sampling temperature.
-   lora_config (str, Optional, defaults to None) -- Set lora config path.
-   adapter_name (str, Optional, defaults to default) -- The name of the adapter to be injected, if not provided, the default adapter name is used ('default').



### class *Merak.MerakTrainer*

> Class of Merak's trainer was derived from *Trainer*  of huggingface transformers for convenience. We provide some argument for user, to support tracing and loss computing

Parameters:

-   leaf_modules (Tuple[`torch.nn.Module`], defaults to ()) -- If a module cannot be traced by `torch.fx`, set it as leaf modules.
-   loss_fn (`torch.nn.Module`, defaults to `torch.nn.CrossEntropyLoss()`) -- Loss function that computes loss value. Merak would not use `trainer.compute_loss`.

    


### *Merak.set_tp_layer_lists*

> Set the tp feature dict for model does not have default dict. Indicates the layers and attributes needs to be changed according to the tp degree. Could refer `Merak.modules.mp_mapping.MP_MODEL_MAPPING` as examples.

Parameters:

-   col_para_list (List[str], defaults to None) -- Name list of linear layer for column parallel style.
-   row_para_list (List[str], defaults to None) -- Name list of linear layer for row parallel style..
-   input_output_mapping (List[str], defaults to None) -- Ratio between input and output of linear layer to indicate the tp style, list of (input, output, 'row' or 'col')
-   weight_change_list (List[str], defaults to None) -- List of (layer name, tp dimension), will divide the tp dimension of layer name or layer_name.weight by the tp degree.
-   tp_attr_list (List[str], defaults to None) -- Manual tp attributes list for each layer, each attribute will be divided by tp degree.
