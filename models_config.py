"""
Maps the five LLM models to the two tokenizer families they use.
K=2 Tokenizer Familieis for Encoder/Decoder Heads
K=5 LLM Adapters for Stage 2
"""
import torch

models = {
    "Qwen/Qwen3-1.7B": {
        "ctx_len": 32768,
        "attn_q": 16,
        "attn_kv": 8,
        "dtype": torch.bfloat16
    },

    "Qwen/Qwen3-4B": {
        "ctx_len": 32768,
        "attn_q": 32,
        "attn_kv": 8,
        "dtype": torch.bfloat16
    },

    "Qwen/Qwen3-8B-FP8": {
        "ctx_len": 32768,
        "attn_q": 32,
        "attn_kv": 8,
        "dtype": torch.float8_e4m3fn
    },









}