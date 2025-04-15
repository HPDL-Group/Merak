def load_config(model_name):
    if model_name == "electra-base":
        config = {
        "architectures": [
            "ElectraForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "embedding_size": 768,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 256,
        "initializer_range": 0.02,
        "intermediate_size": 1024,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "electra",
        "num_attention_heads": 4,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "position_embedding_type": "absolute",
        "summary_activation": "gelu",
        "summary_last_dropout": 0.1,
        "summary_type": "first",
        "summary_use_proj": True,
        "transformers_version": "4.6.0.dev0",
        "type_vocab_size": 2,
        'return_dict': False,
        "use_cache": False,
        "vocab_size": 30522
        }
    else:
        raise ValueError(f"No {model_name} config")
    return config