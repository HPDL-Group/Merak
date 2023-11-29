
def load_config(model_name):
    config = {
        'activation_function': 'gelu',
        'architectures': ['GPT2LMHeadModel'],
        'attn_pdrop': 0.1,
        'bos_token_id': 127999,
        'embd_pdrop': 0.1,
        'eos_token_id': 0,
        'initializer_range': 0.02,
        'layer_norm_epsilon': 1e-05,
        'model_type': 'gpt2',
        "n_ctx": 1024,
        'n_embd': 768,
        'n_head': 12,
        'n_layer': 12,
        'n_positions': 1024,
        'resid_pdrop': 0.1,
        'summary_activation': None,
        'summary_first_dropout': 0.1,
        'summary_proj_to_labels': True,
        'summary_type': 'cls_index',
        'summary_use_proj': True,
        'task_specific_params': {'text-generation': {'do_sample': True, 'max_length': 50}},
        'vocab_size': 50304,
        'return_dict': False,
        'reorder_and_upcast_attn': True,
        'use_cache': False
        }
    return config
