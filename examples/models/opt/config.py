def load_config(model_name):
    if model_name == "opt-350m":
        config = {
            "_name_or_path": "opt-350m",
            "activation_dropout": 0.0,
            "activation_function": "relu",
            "architectures": [
                "OPTForCausalLM"
            ],
            "attention_dropout": 0.0,
            "bos_token_id": 2,
            "do_layer_norm_before": False,
            "dropout": 0.1,
            "eos_token_id": 2,
            "ffn_dim": 4096,
            "hidden_size": 1024,
            "init_std": 0.02,
            "layerdrop": 0.0,
            "max_position_embeddings": 2048,
            "model_type": "opt",
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "pad_token_id": 1,
            "prefix": "</s>",
            "torch_dtype": "float16",
            "transformers_version": "4.20.0.dev0",
            "use_cache": False,
            'return_dict': False,
            "vocab_size": 50272,
            "word_embed_proj_dim": 512
            }
    else:
        raise ValueError(f"No {model_name} config")
    return config