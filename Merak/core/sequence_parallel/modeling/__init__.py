from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from .llama import LlamaDecoderLayerSP

MODELS_SP_CONFIG = {
    "Llama" : {
        "models" :(   
            "LlamaModel", 
            "LlamaForCausalLM",
            "LlamaForSequenceClassification",
            "LlamaForQuestionAnswering",
            "LlamaForTokenClassification",
        ),
        "sub_module_replacement": {
            LlamaDecoderLayer: LlamaDecoderLayerSP
        },
    },
}


__all__ = [
    'MODELS_SP_CONFIG',
]
