def load_config(model_name):
    if model_name == "gpt-j-6b":
        config = {
        "activation_function": "gelu_new",
        "architectures": [
            "GPTJForCausalLM"
        ],
        "attn_pdrop": 0.0,
        "bos_token_id": 50256,
        "embd_pdrop": 0.0,
        "eos_token_id": 50256,
        "gradient_checkpointing": False,
        "initializer_range": 0.02,
        "layer_norm_epsilon": 1e-05,
        "model_type": "gptj",
        "n_embd": 4096,
        "n_head": 16,
        "n_inner": None,
        "n_layer": 4, #28,
        "n_positions": 2048,
        "resid_pdrop": 0.0,
        "rotary": True,
        "rotary_dim": 64,
        "scale_attn_weights": True,
        "summary_activation": None,
        "summary_first_dropout": 0.1,
        "summary_proj_to_labels": True,
        "summary_type": "cls_index",
        "summary_use_proj": True,
        "task_specific_params": {
            "text-generation": {
            "do_sample": True,
            "max_length": 50,
            "temperature": 1.0
            }
        },
        "tie_word_embeddings": False,
        "tokenizer_class": "GPT2Tokenizer",
        "transformers_version": "4.18.0.dev0",
        'return_dict': False,
        "use_cache": False,
        "vocab_size": 50400
        }
    else:
        raise ValueError(f"No {model_name} config")
    return config