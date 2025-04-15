def load_config(model_name):
    if model_name == "hubert-base-ls960":
        config = {
            "_name_or_path": "facebook/hubert-base-ls960",
            "activation_dropout": 0.1,
            "apply_spec_augment": True,
            "architectures": [
                "HubertModel"
            ],
            "attention_dropout": 0.1,
            "bos_token_id": 1,
            "conv_bias": False,
            "conv_dim": [
                512,
                512,
                512,
                512,
                512,
                512,
                512
            ],
            "conv_kernel": [
                10,
                3,
                3,
                3,
                3,
                2,
                2
            ],
            "conv_stride": [
                5,
                2,
                2,
                2,
                2,
                2,
                2
            ],
            "ctc_loss_reduction": "sum",
            "ctc_zero_infinity": False,
            "do_stable_layer_norm": False,
            "eos_token_id": 2,
            "feat_extract_activation": "gelu",
            "feat_extract_dropout": 0.0,
            "feat_extract_norm": "group",
            "feat_proj_dropout": 0.1,
            "final_dropout": 0.1,
            "gradient_checkpointing": False,
            "hidden_act": "gelu",
            "hidden_dropout": 0.1,
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-05,
            "layerdrop": 0.1,
            "mask_feature_length": 10,
            "mask_feature_prob": 0.0,
            "mask_time_length": 10,
            "mask_time_prob": 0.05,
            "model_type": "hubert",
            "num_attention_heads": 12,
            "num_conv_pos_embedding_groups": 16,
            "num_conv_pos_embeddings": 128,
            "num_feat_extract_layers": 7,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "transformers_version": "4.10.0.dev0",
            "vocab_size": 32,
            'return_dict': False,
            "use_cache": False,
            "tokenizer_class": "Wav2Vec2CTCTokenizer"
            }
    else:
        raise ValueError(f"No {model_name} config")
    return config