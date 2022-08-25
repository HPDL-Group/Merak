# Mixed Precision Training

Merak supports automatic mixed precision (amp) training and fully fp16 training.

## Automatic mixed precision training

1. Please ensure apex is installed. Installation please refer to [apex repository](https://github.com/NVIDIA/apex).

2. Enable amp training with setting `half_precision_backend='apex'` in `Merak.MerakArguments`. Detail usage of this config can be found in [transformers trainer arguments](https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/trainer#transformers.TrainingArguments).

3. Amp training only supports O1 level, DP, and PMP for now.


## FP16 training


Enable fp16 training with setting `fp16=true` in `Merak.MerakArguments`. Detail usage of this config can be found in [transformers trainer arguments](https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/trainer#transformers.TrainingArguments).

More configuration of fp16 training, including `loss_scale`, `initial_scale_power`, `loss_scale_window`, `hysteresis`, and `min_loss_scale`, can be setted with `Merak.MerakArguments`. Detail usages of these configs please refer to our api [document](https://github.com/HPDL-Group/Merak/blob/main/docs/api_doc.md).