def load_config(model_name):
    if model_name == "t5-base":
        config = {
            'architectures': ['T5WithLMHeadModel'], 
            'd_ff': 3072, 
            'd_kv': 64, 
            'd_model': 512, 
            'decoder_start_token_id': 0, 
            'dropout_rate': 0.1, 
            'eos_token_id': 1, 
            'initializer_factor': 1.0, 
            'is_encoder_decoder': True, 
            'layer_norm_epsilon': 1e-06, 
            'model_type': 't5', 
            'n_positions': 512, 
            'num_heads': 16, 
            'num_layers': 8, 
            'output_past': True, 
            'pad_token_id': 0, 
            'relative_attention_num_buckets': 32, 
            'task_specific_params': {'summarization': {'early_stopping': True, 'length_penalty': 2.0, 'max_length': 200, 'min_length': 30, 'no_repeat_ngram_size': 3, 'num_beams': 4, 'prefix': 'summarize: '}, 
            'translation_en_to_de': {'early_stopping': True, 'max_length': 300, 'num_beams': 4, 'prefix': 'translate English to German: '}, 
            'translation_en_to_fr': {'early_stopping': True, 'max_length': 300, 'num_beams': 4, 
            'prefix': 'translate English to French: '}, 
            'translation_en_to_ro': {'early_stopping': True, 'max_length': 300, 
            'num_beams': 4, 
            'prefix': 'translate English to Romanian: '}}, 
            'vocab_size': 32128,
            'use_cache': False,
            'return_dict': False,
            }
    return config