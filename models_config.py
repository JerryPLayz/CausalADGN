"""
Maps the five LLM models to the two tokenizer families they use.
K=2 Tokenizer Familieis for Encoder/Decoder Heads
K=5 LLM Adapters for Stage 2
"""
import torch

models = {
    # Qwen3
    "Qwen/Qwen3-1.7B": {
        "ctx_len": 32768,
        "attn_q": 16,
        "attn_kv": 8,
        "dtype": torch.bfloat16,
        "link": "https://huggingface.co/Qwen/Qwen3-1.7B",
        "license": False,
        "exceeds": False,
    },

    "Qwen/Qwen3-4B": {
        "ctx_len": 32768,
        "attn_q": 32,
        "attn_kv": 8,
        "dtype": torch.bfloat16,
        "link": "https://huggingface.co/Qwen/Qwen3-4B",
        "license": False,
        "exceeds": False,

    },

    "Qwen/Qwen3-8B-FP8": {
        "ctx_len": 32768,
        "attn_q": 32,
        "attn_kv": 8,
        "dtype": torch.float8_e4m3fn,
        "link": "https://huggingface.co/Qwen/Qwen3-8B-FP8",
        "license": False,
        "exceeds": False,
    },

    # Llama-3.2
    "meta-llama/Llama-3.2-1B": {
        "ctx_len": 128000,
        "attn_q": 0,
        "attn_kv": 0,
        "dtype": torch.bfloat16,
        "link": "https://huggingface.co/meta-llama/Llama-3.2-1B",
        "license": True,
        "exceeds": False,
    },

    "meta-llama/Llama-3.2-3B": {
        "ctx_len": 128000,
        "attn_q": 0,
        "attn_kv": 0,
        "dtype": torch.bfloat16,
        "link": "https://huggingface.co/meta-llama/Llama-3.2-3B",
        "license": True,
        "exceeds": False,

    },

    "meta-llama/Llama-3.1-8B": {
        "ctx_len": 128000,
        "attn_q": 0,
        "attn_kv": 0,
        "dtype": torch.bfloat16,
        "license": True,
        "exceeds": True,
    }

    # Other Families...














}