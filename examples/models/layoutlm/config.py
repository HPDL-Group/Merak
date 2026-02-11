def load_config(model_name):
    if model_name == "layoutlm-base-uncased":
        config = {
            "_name_or_path": "microsoft/layoutlm-base-uncased",
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_2d_position_embeddings": 1024,
            "max_position_embeddings": 512,
            "model_type": "layoutlm",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "output_past": True,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.4.0.dev0",
            "type_vocab_size": 2,
            "use_cache": True,
            "return_dict": False,
            "vocab_size": 30522,
        }
    else:
        raise ValueError(f"No {model_name} config")
    return config
