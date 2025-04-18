def load_config(model_name):
    if model_name == "opus-mt-zh-en":
        config = {
            "_name_or_path": "/tmp/Helsinki-NLP/opus-mt-zh-en",
            "activation_dropout": 0.0,
            "activation_function": "swish",
            "add_bias_logits": False,
            "add_final_layer_norm": False,
            "architectures": [
                "MarianMTModel"
            ],
            "attention_dropout": 0.0,
            "bad_words_ids": [
                [
                65000
                ]
            ],
            "bos_token_id": 0,
            "classif_dropout": 0.0,
            "classifier_dropout": 0.0,
            "d_model": 512,
            "decoder_attention_heads": 8,
            "decoder_ffn_dim": 2048,
            "decoder_layerdrop": 0.0,
            "decoder_layers": 6,
            "decoder_start_token_id": 65000,
            "decoder_vocab_size": 65001,
            "dropout": 0.1,
            "encoder_attention_heads": 8,
            "encoder_ffn_dim": 2048,
            "encoder_layerdrop": 0.0,
            "encoder_layers": 6,
            "eos_token_id": 0,
            "extra_pos_embeddings": 65001,
            "forced_eos_token_id": 0,
            "id2label": {
                "0": "LABEL_0",
                "1": "LABEL_1",
                "2": "LABEL_2"
            },
            "init_std": 0.02,
            "is_encoder_decoder": True,
            "label2id": {
                "LABEL_0": 0,
                "LABEL_1": 1,
                "LABEL_2": 2
            },
            "max_length": 512,
            "max_position_embeddings": 512,
            "model_type": "marian",
            "normalize_before": False,
            "normalize_embedding": False,
            "num_beams": 6,
            "num_hidden_layers": 6,
            "pad_token_id": 65000,
            "scale_embedding": True,
            "share_encoder_decoder_embeddings": True,
            "static_position_embeddings": True,
            "transformers_version": "4.22.0.dev0",
            "use_cache": False,
            'return_dict': False,
            "vocab_size": 65001
            }
    else:
        raise ValueError(f"No {model_name} config")
    return config