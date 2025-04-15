def load_config(model_name):
    if model_name == "mt5-base":
        config = {
            "_name_or_path": "/home/patrick/hugging_face/t5/mt5-base",
            "architectures": [
                "MT5ForConditionalGeneration"
            ],
            "d_ff": 2048,
            "d_kv": 64,
            "d_model": 768,
            "decoder_start_token_id": 0,
            "dropout_rate": 0.1,
            "eos_token_id": 1,
            "feed_forward_proj": "gated-gelu",
            "initializer_factor": 1.0,
            "is_encoder_decoder": True,
            "layer_norm_epsilon": 1e-06,
            "model_type": "mt5",
            "num_decoder_layers": 12,
            "num_heads": 12,
            "num_layers": 12,
            "output_past": True,
            "pad_token_id": 0,
            "relative_attention_num_buckets": 32,
            "tie_word_embeddings": False,
            "tokenizer_class": "T5Tokenizer",
            "transformers_version": "4.10.0.dev0",
            "use_cache": False,
            "vocab_size": 250112
            }
    else:
        raise ValueError(f"No {model_name} config")
    return config