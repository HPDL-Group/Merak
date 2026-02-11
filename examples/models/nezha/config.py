def load_config(model_name):
    if model_name == "nezha-cn-base":
        config = {
            "_name_or_path": "nezha-cn-base",
            "architectures": ["NeZhaForMaskedLM"],
            "attention_probs_dropout_prob": 0.1,
            "bos_token_id": 2,
            "classifier_dropout": 0.1,
            "embedding_size": 128,
            "eos_token_id": 3,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "inner_group_num": 1,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "max_relative_position": 64,
            "model_type": "nezha",
            "num_attention_heads": 12,
            "num_hidden_groups": 1,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "torch_dtype": "float32",
            "transformers_version": "4.20.0.dev0",
            "type_vocab_size": 2,
            "use_cache": False,
            "use_relative_position": True,
            "return_dict": False,
            "vocab_size": 21128,
        }
    else:
        raise ValueError(f"No {model_name} config")
    return config
