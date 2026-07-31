"""
Maps the five LLM models to the two tokenizer families they use.
K=2 Tokenizer Familieis for Encoder/Decoder Heads
K=5 LLM Adapters for Stage 2
"""

TOKENIZER_FAMILIES = {
    "qwen3": {
        "model_name_or_path": "Qwen/Qwen3-1.7B",
        "vocab_size": 151936,
    },
    "llama3": {
        "model_name_or_path": "meta-llama/Llama-3.2-1B",
        "vocab_size": 128256,
    },
}

LLM_ADAPTERS = {
    # Qwen Models
    "qwen3_1b7": {
        "family":           "qwen3",
        "model_name":       "Qwen/Qwen3-1.7B",
        "llm_hidden_size":  2048
    },
    "qwen3_8b": {
        "family":           "qwen3",
        "model_name":       "Qwen/Qwen3-8B",
        "llm_hidden_size":  4096,
    },

    # Llama Models
    "llama3_1b": {
        "family":           "llama3",
        "model_name":       "meta-llama/Llama-3.2-1B",
        "llm_hidden_size":  2048,
    },
    "llama3_3b": {
        "family":           "llama3",
        "model_name":       "meta-llama/Llama-3.2-3B",
        "llm_hidden_size":  3072,
    },
    "llama3_8b": {
        "family":           "llama3",
        "model_name":       "meta-llama/Llama-3.2-8B",
        "llm_hidden_size":  4096,
    },

}