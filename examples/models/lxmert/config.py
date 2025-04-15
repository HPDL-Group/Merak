def load_config(model_name):
    if model_name == "lxmert-vqa-uncased":
        config = {
            "architectures": [
                "LxmertForQuestionAnswering"
            ],
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "l_layers": 9,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "lxmert",
            "num_attention_heads": 12,
            "num_attr_labels": 400,
            "num_hidden_layers": {
                "cross_encoder": 5,
                "language": 9,
                "vision": 5
            },
            "num_object_labels": 1600,
            "num_qa_labels": 3129,
            "r_layers": 5,
            "task_mask_lm": True,
            "task_matched": True,
            "task_obj_predict": True,
            "task_qa": True,
            "type_vocab_size": 2,
            "visual_attr_loss": True,
            "visual_feat_dim": 2048,
            "visual_feat_loss": True,
            "visual_loss_normalizer": 6.67,
            "visual_obj_loss": True,
            "visual_pos_dim": 4,
            "vocab_size": 30522,
            'return_dict': False,
            "use_cache": False,
            "x_layers": 5
            }
    else:
        raise ValueError(f"No {model_name} config")
    return config