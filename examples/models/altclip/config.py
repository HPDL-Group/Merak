def load_config(model_name):
    if model_name == "AltCLIP":
        config = {
        "_commit_hash": "4d06f38b304fc2a331d9f3eab77a542afafc4ffb",
        "_name_or_path": "BAAI/AltCLIP",
        "architectures": [
            "AltCLIPModel"
        ],
        "direct_kd": False,
        "initializer_factor": 1.0,
        "logit_scale_init_value": 2.6592,
        "model_type": "altclip",
        "num_layers": 3,
        "projection_dim": 768,
        "text_config": {
            "_name_or_path": "",
            "add_cross_attention": False,
            "architectures": None,
            "attention_probs_dropout_prob": 0.1,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": 0,
            "chunk_size_feed_forward": 0,
            "classifier_dropout": None,
            "cross_attention_hidden_size": None,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "early_stopping": False,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": 2,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 1024,
            "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
            },
            "initializer_factor": 0.02,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
            },
            "layer_norm_eps": 1e-05,
            "length_penalty": 1.0,
            "max_length": 20,
            "max_position_embeddings": 514,
            "min_length": 0,
            "model_type": "altclip_text_model",
            "no_repeat_ngram_size": 0,
            "num_attention_heads": 16,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_hidden_layers": 24,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": 1,
            "pooler_fn": "cls",
            "position_embedding_type": "absolute",
            "prefix": None,
            "problem_type": None,
            "project_dim": 768,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "sep_token_id": None,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": True,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": None,
            "torchscript": False,
            "transformers_version": "4.26.0.dev0",
            "type_vocab_size": 1,
            "typical_p": 1.0,
            "use_bfloat16": False,
            "use_cache": True,
            "vocab_size": 250002
        },
        "text_config_dict": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 24
        },
        "text_model_name": None,
        "torch_dtype": "float32",
        "transformers_version": None,
        "vision_config": {
            "_name_or_path": "",
            "add_cross_attention": False,
            "architectures": None,
            "attention_dropout": 0.0,
            "bad_words_ids": None,
            "begin_suppress_tokens": None,
            "bos_token_id": None,
            "chunk_size_feed_forward": 0,
            "cross_attention_hidden_size": None,
            "decoder_start_token_id": None,
            "diversity_penalty": 0.0,
            "do_sample": False,
            "dropout": 0.0,
            "early_stopping": False,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": None,
            "exponential_decay_length_penalty": None,
            "finetuning_task": None,
            "forced_bos_token_id": None,
            "forced_eos_token_id": None,
            "hidden_act": "quick_gelu",
            "hidden_size": 1024,
            "id2label": {
            "0": "LABEL_0",
            "1": "LABEL_1"
            },
            "image_size": 224,
            "initializer_factor": 1.0,
            "initializer_range": 0.02,
            "intermediate_size": 4096,
            "is_decoder": False,
            "is_encoder_decoder": False,
            "label2id": {
            "LABEL_0": 0,
            "LABEL_1": 1
            },
            "layer_norm_eps": 1e-05,
            "length_penalty": 1.0,
            "max_length": 20,
            "min_length": 0,
            "model_type": "altclip_vision_model",
            "no_repeat_ngram_size": 0,
            "num_attention_heads": 16,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_channels": 3,
            "num_hidden_layers": 24,
            "num_return_sequences": 1,
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "pad_token_id": None,
            "patch_size": 14,
            "prefix": None,
            "problem_type": None,
            "projection_dim": 512,
            "pruned_heads": {},
            "remove_invalid_values": False,
            "repetition_penalty": 1.0,
            "return_dict": True,
            "return_dict_in_generate": False,
            "sep_token_id": None,
            "suppress_tokens": None,
            "task_specific_params": None,
            "temperature": 1.0,
            "tf_legacy_loss": False,
            "tie_encoder_decoder": False,
            "tie_word_embeddings": True,
            "tokenizer_class": None,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": None,
            "torchscript": False,
            "transformers_version": "4.26.0.dev0",
            "typical_p": 1.0,
            "use_bfloat16": False
        },
        "vision_config_dict": {
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "patch_size": 14
        },
        "vision_model_name": None
        }
    else:
        raise ValueError(f"No {model_name} config")
    return config