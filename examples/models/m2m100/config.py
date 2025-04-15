def load_config(model_name):
    if model_name == "m2m100_418M":
        config = {
            "_name_or_path": "hf_models/m2m100_418M",
            "activation_dropout": 0.0,
            "activation_function": "relu",
            "architectures": [
                "M2M100ForConditionalGeneration"
            ],
            "attention_dropout": 0.1,
            "bos_token_id": 0,
            "d_model": 1024,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": 4096,
            "decoder_layerdrop": 0.05,
            "decoder_layers": 4,
            "decoder_start_token_id": 2,
            "dropout": 0.1,
            "early_stopping": True,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "encoder_layerdrop": 0.05,
            "encoder_layers": 4,
            "eos_token_id": 2,
            "gradient_checkpointing": False,
            "init_std": 0.02,
            "is_encoder_decoder": True,
            "max_length": 200,
            "max_position_embeddings": 1024,
            "model_type": "m2m_100",
            "num_beams": 5,
            "num_hidden_layers": 4,
            "pad_token_id": 1,
            "scale_embedding": True,
            "transformers_version": "4.4.0.dev0",
            'return_dict': False,
            "use_cache": False,
            "vocab_size": 128112
            }
    else:
        raise ValueError(f"No {model_name} config")
    return config