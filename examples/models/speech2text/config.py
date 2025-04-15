def load_config(model_name):
    if model_name == "s2t-medium-librispeech-asr":
        config = {
            "_name_or_path": "hf_models_fb/s2t-medium-librispeech-asr/",
            "activation_dropout": 0.15,
            "activation_function": "relu",
            "architectures": [
                "Speech2TextForConditionalGeneration"
            ],
            "attention_dropout": 0.15,
            "bos_token_id": 0,
            "classifier_dropout": 0.0,
            "conv_channels": 1024,
            "conv_kernel_sizes": [
                5,
                5
            ],
            "d_model": 512,
            "decoder_attention_heads": 8,
            "decoder_ffn_dim": 2048,
            "decoder_layerdrop": 0.0,
            "decoder_layers": 6,
            "decoder_start_token_id": 2,
            "dropout": 0.15,
            "early_stopping": True,
            "encoder_attention_heads": 8,
            "encoder_ffn_dim": 2048,
            "encoder_layerdrop": 0.0,
            "encoder_layers": 12,
            "eos_token_id": 2,
            "gradient_checkpointing": False,
            "init_std": 0.02,
            "input_channels": 1,
            "input_feat_per_channel": 80,
            "is_encoder_decoder": True,
            "max_length": 200,
            "max_source_positions": 6000,
            "max_target_positions": 1024,
            "model_type": "speech_to_text",
            "num_beams": 5,
            "num_conv_layers": 2,
            "num_hidden_layers": 12,
            "pad_token_id": 1,
            "scale_embedding": True,
            "transformers_version": "4.4.0.dev0",
            "use_cache": False,
            'return_dict': False,
            "vocab_size": 10000
            }
    else:
        raise ValueError(f"No {model_name} config")
    return config