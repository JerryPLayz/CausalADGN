import gc

import env
import torch
from CADGNTrainer import CADGNTrainer, Stage1Config, Stage2Config

from models_config import models
from ds.cladder import CLadderSample, CLadderDataset, load_cladder_v1_5, CLadderLoaderConfig
from cadgn import CADGNCore, TokenizerFamily, flush_gpu, save_history, BaseEncoder, LLMWrapper, eval_and_save
from search import ArchParams
import gc

# Unfortunately, the current implementation precludes saving modified versions of the various modules (config is static)

flush_gpu()
gc.collect()

epochs = 50

do_models = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
]

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
    w_mmd = 1.0,
    epochs=epochs
)

evals2c = Stage2Config(
  **s2c.__dict__
)
evals2c.max_steps_per_epoch_val = None
evals2c.max_steps_per_epoch = None

dsConfig = CLadderLoaderConfig(rung_filter=None, query_types=None, skip_unparseable=True)

org_ds = load_cladder_v1_5(dsConfig)

train, vald = CLadderDataset.from_samples_split(
    samples=org_ds,
    val_size=0.2,
    stratify=True
)

train_d = train.as_dataloader(max_steps_per_epoch=550)
val_d = vald.as_dataloader(max_steps_per_epoch=150, shuffle=False)

complete_val = vald.as_dataloader(shuffle=False)
prompt_samples: list[CLadderSample] = CLadderDataset(org_ds).stratified_sample(50)



import numpy as np
import pandas as pd

# from cadgn import GateClassifier

# from ablations.adgn_encoder import ADGNEncoder
# from ablations.gat_encoder import GATEncoder
# from ablations.gcn_encoder import GCNEncoder
from ablations.BaselineLLMTrainer import BaselineLLMTrainer, BaselineSampleResult

# Baselines (simplest, no cross-family matters)
for mdl in do_models:
    mdl: str = mdl
    print(f"Establishing baseline for LLM {mdl}")
    with LLMWrapper(mdl) as llm:
        mmd_val = 0.0  # MMD doesn't apply here as the projectors aren't involved.
        s2c.w_mmd = mmd_val
        evals2c.w_mmd = mmd_val
        tokfam = TokenizerFamily.from_llm(
            llm,
            max_seq_len=128,
            **arch.family_kwargs()
        )
        blt = BaselineLLMTrainer(tokfam.gate_classifier, device="cuda")

        history, train_per_sample = blt.train(s2c, tokfam, train_dataloader=train_d, val_dataloader=val_d, llm=llm)

        yes_ids, no_ids = llm.get_yn_token_sets(tokfam.tokenizer)
        metrics, per_sample = blt._eval(
            val_dataloader=complete_val,
            config=evals2c,
            family=tokfam,
            yes_ids=yes_ids,
            no_ids=no_ids,
            llm=llm,
        )
        eval_and_save(
            llm=llm,
            mmd_val=mmd_val,
            tokfam=tokfam,
            history=history,
            metrics=metrics,
            per_sample=per_sample,
            cls="baseline",
            prompt_samples=prompt_samples,
        )
    print("Deconstructing model...")
    del tokfam
    del blt
    del history, train_per_sample
    del metrics, per_sample
    del yes_ids, no_ids
    flush_gpu()
    gc.collect()


