def load_config(model_name):
    if model_name == "distilbert-base-cased":
        config = {
            "activation": "gelu",
            "architectures": [
                "DistilBertForMaskedLM"
            ],
            "attention_dropout": 0.1,
            "dim": 768,
            "dropout": 0.1,
            "hidden_dim": 3072,
            "initializer_range": 0.02,
            "max_position_embeddings": 512,
            "model_type": "distilbert",
            "n_heads": 12,
            "n_layers": 6,
            "output_past": True,
            "pad_token_id": 0,
            "qa_dropout": 0.1,
            "seq_classif_dropout": 0.2,
            "sinusoidal_pos_embds": False,
            "tie_weights_": True,
            'return_dict': False,
            "use_cache": False,
            "_attn_implementation": 'eager',
            "vocab_size": 28996
        }
    else:
        raise ValueError(f"No {model_name} config")
    return config