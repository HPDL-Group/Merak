def load_config(model_name):
    if model_name == "bart-large-mnli":
        config = {
        "_num_labels": 3,
        "activation_dropout": 0.0,
        "activation_function": "gelu",
        "add_final_layer_norm": False,
        "architectures": [
            "BartForSequenceClassification"
        ],
        "_attn_implementation": 'eager',
        "attention_dropout": 0.0,
        "bos_token_id": 0,
        "classif_dropout": 0.0,
        "classifier_dropout": 0.0,
        "d_model": 1024,
        "decoder_attention_heads": 16,
        "decoder_ffn_dim": 4096,
        "decoder_layerdrop": 0.0,
        "decoder_layers": 12,
        "decoder_start_token_id": 2,
        "dropout": 0.1,
        "encoder_attention_heads": 16,
        "encoder_ffn_dim": 4096,
        "encoder_layerdrop": 0.0,
        "encoder_layers": 12,
        "eos_token_id": 2,
        "forced_eos_token_id": 2,
        "gradient_checkpointing": False,
        "id2label": {
            "0": "contradiction",
            "1": "neutral",
            "2": "entailment"
        },
        "init_std": 0.02,
        "is_encoder_decoder": True,
        "label2id": {
            "contradiction": 0,
            "entailment": 2,
            "neutral": 1
        },
        "max_position_embeddings": 1024,
        "model_type": "bart",
        "normalize_before": False,
        "num_hidden_layers": 12,
        "output_past": False,
        "pad_token_id": 1,
        "scale_embedding": False,
        "transformers_version": "4.7.0.dev0",
        "use_cache": False,
        "vocab_size": 50265
        }
    return config