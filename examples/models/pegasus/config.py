def load_config(model_name):
    if model_name == "pegasus-large":
        config = {
            "_name_or_path": "google/pegasus-large",
            "activation_dropout": 0.1,
            "activation_function": "relu",
            "add_bias_logits": False,
            "add_final_layer_norm": True,
            "architectures": [
                "PegasusForConditionalGeneration"
            ],
            "attention_dropout": 0.1,
            "bos_token_id": 0,
            "classif_dropout": 0.0,
            "classifier_dropout": 0.0,
            "d_model": 1024,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": 4096,
            "decoder_layerdrop": 0.0,
            "decoder_layers": 16,
            "decoder_start_token_id": 0,
            "dropout": 0.1,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "encoder_layerdrop": 0.0,
            "encoder_layers": 16,
            "eos_token_id": 1,
            "extra_pos_embeddings": 1,
            "force_bos_token_to_be_generated": False,
            "forced_eos_token_id": 1,
            "gradient_checkpointing": False,
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
            "length_penalty": 0.8,
            "max_length": 256,
            "max_position_embeddings": 1024,
            "model_type": "pegasus",
            "normalize_before": True,
            "normalize_embedding": False,
            "num_beams": 8,
            "num_hidden_layers": 16,
            "pad_token_id": 0,
            "scale_embedding": True,
            "static_position_embeddings": True,
            "task_specific_params": {
                "summarization_aeslc": {
                "length_penalty": 0.6,
                "max_length": 32,
                "max_position_embeddings": 512
                },
                "summarization_arxiv": {
                "length_penalty": 0.8,
                "max_length": 256,
                "max_position_embeddings": 1024
                },
                "summarization_big_patent": {
                "length_penalty": 0.7,
                "max_length": 256,
                "max_position_embeddings": 1024
                },
                "summarization_billsum": {
                "length_penalty": 0.6,
                "max_length": 256,
                "max_position_embeddings": 1024
                },
                "summarization_cnn_dailymail": {
                "length_penalty": 0.8,
                "max_length": 128,
                "max_position_embeddings": 1024
                },
                "summarization_gigaword": {
                "length_penalty": 0.6,
                "max_length": 32,
                "max_position_embeddings": 128
                },
                "summarization_large": {
                "length_penalty": 0.8,
                "max_length": 256,
                "max_position_embeddings": 1024
                },
                "summarization_multi_news": {
                "length_penalty": 0.8,
                "max_length": 256,
                "max_position_embeddings": 1024
                },
                "summarization_newsroom": {
                "length_penalty": 0.8,
                "max_length": 128,
                "max_position_embeddings": 512
                },
                "summarization_pubmed": {
                "length_penalty": 0.8,
                "max_length": 256,
                "max_position_embeddings": 1024
                },
                "summarization_reddit_tifu": {
                "length_penalty": 0.6,
                "max_length": 128,
                "max_position_embeddings": 512
                },
                "summarization_wikihow": {
                "length_penalty": 0.6,
                "max_length": 256,
                "max_position_embeddings": 512
                },
                "summarization_xsum": {
                "length_penalty": 0.8,
                "max_length": 64,
                "max_position_embeddings": 512
                }
            },
            "transformers_version": "4.11.0.dev0",
            "use_cache": False,
            'return_dict': False,
            "vocab_size": 96103
            }
    else:
        raise ValueError(f"No {model_name} config")
    return config