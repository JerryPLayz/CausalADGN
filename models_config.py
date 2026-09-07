"""
Maps the five LLM models to the two tokenizer families they use.
K=2 Tokenizer Familieis for Encoder/Decoder Heads
K=5 LLM Adapters for Stage 2
"""
import torch
from CADGNTrainer import Stage1Config, Stage2Config
from search import ArchParams

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
        #"dtype": torch.float8_e4m3fn,
        "dtype": torch.bfloat16,
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

    # Llama-3.1
    "meta-llama/Llama-3.1-8B": {
        "ctx_len": 128000,
        "attn_q": 0,
        "attn_kv": 0,
        "dtype": torch.bfloat16,
        "link": "https://huggingface.co/meta-llama/Llama-3.1-8B",
        "license": True,
        "exceeds": True,
    }

    # Other Families...

}

## Other configuration
# rounded to 4dp where appropriate
best_params = {
    "ca_dgn_dim": 256,
    "max_layers": 5,
    "num_iters": 4,
    "epsilon": 0.0665, # 0.06646303815100524
    "base_gamma": 0.0296, # 0.02956587470709161
    "encoder_dropout": 0.0563,  # 0.05627028512612424
    "decoder_expansion": 4,
    "decoder_dropout": 0.1743,  # 0.17434218753510897
    "head_encoder_dropout": 0.0595,  # 0.05948165802005359
    "head_decoder_dropout": 0.0543,  # 0.0543472754267514
}

arch = ArchParams.from_dict(best_params)


def get_stage_configs(epochs) -> tuple[Stage1Config, Stage2Config, Stage2Config]:
    """
    Constructs related Stage 1 & Stage 2 configs
    :param epochs:
    :return:
    """
    s1c = Stage1Config(
        lr=0.0015,  # 0.0015121612751471838
        weight_decay=0.0004,  # 0.0004379467905083944
        grad_accum_steps=13,
        # Default
        max_steps_per_epoch=550,
        max_steps_per_epoch_val=150,
        w_mmd=1.0,
        epochs=epochs
    )
    s2c = Stage2Config(
        lr=0.0015,  # 0.0015121612751471838
        weight_decay=0.0004,  # 0.0004379467905083944
        grad_accum_steps=13,
        max_steps_per_epoch=550,
        max_steps_per_epoch_val=150,
        w_mmd=1.0,
        epochs=epochs
    )
    evals2c = Stage2Config(
        **s2c.__dict__
    )
    evals2c.max_steps_per_epoch_val = None
    evals2c.max_steps_per_epoch = None
    return s1c, s2c, evals2c

