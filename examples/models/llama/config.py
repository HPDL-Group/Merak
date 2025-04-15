def load_config(model_name):
    if model_name == "llama":
        config = {"architectures": ["LlamaForCausalLM"], 
                "bos_token_id": 0, 
                "eos_token_id": 1, 
                "hidden_act": "silu", 
                "hidden_size": 4096, 
                "intermediate_size": 11008, 
                "initializer_range": 0.02, 
                "max_sequence_length": 2048, 
                "model_type": "llama", 
                "num_attention_heads": 32, 
                "num_hidden_layers": 32, 
                "pad_token_id": -1, 
                "rms_norm_eps": 1e-06, 
                "use_cache": True, 
                "vocab_size": 32000,
                "return_dict": False,
                "use_cache": False,
                }
    else:
        raise ValueError(f"No {model_name} config")
    return config
