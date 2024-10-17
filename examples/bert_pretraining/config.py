def load_config(model_name):
    if model_name == "bert-large":
        config = {
            'architectures': ['BertForPreTraining'], 
            'attention_probs_dropout_prob': 0.1, 
            'gradient_checkpointing': False, 
            'hidden_act': 'gelu', 
            'hidden_dropout_prob': 0.1, 
            'hidden_size': 1024, 
            'initializer_range': 0.02, 
            'intermediate_size': 4096, 
            'layer_norm_eps': 1e-12, 
            'max_position_embeddings': 512, 
            'model_type': 'bert', 
            'num_attention_heads': 16, 
            'num_hidden_layers': 24, 
            'pad_token_id': 0, 
            'position_embedding_type': 'absolute', 
            'type_vocab_size': 2, 
            '_attn_implementation': 'eager',
            'use_cache': True, 
            'return_dict': False,
            'vocab_size': 30522,
            } 
    else:
        raise ValueError(f"No {model_name} config")

    return config
