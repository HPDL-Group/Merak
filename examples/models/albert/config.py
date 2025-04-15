def load_config(model_name):
    if model_name == "albert-base-v1":
        config = {
        "architectures": [
            "AlbertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "bos_token_id": 2,
        "classifier_dropout_prob": 0.1,
        "down_scale_factor": 1,
        "embedding_size": 128,
        "eos_token_id": 3,
        "gap_size": 0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "inner_group_num": 1,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "albert",
        "net_structure_type": 0,
        "num_attention_heads": 12,
        "num_hidden_groups": 1,
        "num_hidden_layers": 12,
        "num_memory_blocks": 0,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30000
        }
    return config