def load_config(model_name):
    if model_name == "dinov2-base":
        config = {
            "architectures": [
                "Dinov2Model"
            ],
            "attention_probs_dropout_prob": 0.0,
            "drop_path_rate": 0.0,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.0,
            "hidden_size": 768,
            "image_size": 518,
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-06,
            "layerscale_value": 1.0,
            "mlp_ratio": 4,
            "model_type": "dinov2",
            "num_attention_heads": 12,
            "num_channels": 3,
            "num_hidden_layers": 12,
            "patch_size": 14,
            "qkv_bias": True,
            "torch_dtype": "float32",
            "transformers_version": "4.31.0.dev0",
            'return_dict': False,
            "use_cache": True,
            "use_swiglu_ffn": False
            }
    else:
        raise ValueError(f"No {model_name} config")
    return config