import gc

import env
import torch
import torch.nn as nn
from CADGNTrainer import CADGNTrainer, Stage1Config, Stage2Config

from models_config import models
from ds.cladder import CLadderSample, CLadderDataset, load_cladder_v1_5, CLadderLoaderConfig
from cadgn import flush_gpu
from search import ArchParams
import gc

from typing import Union, Any
from pathlib import Path
import json
from collections import defaultdict
from sample_tensors import b64_to_tensor, compact_tensor_to_b64, safe_convert_tensor
from InProgressInstance import InProgressInstance, gather_instances
from models_config import get_stage_configs, arch

# Unfortunately, the current implementation precludes saving modified versions of the various modules (config is static)

flush_gpu()
gc.collect()

epochs = 40
s1c, s2c, evals2c = get_stage_configs(epochs)

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

import numpy as np
import pandas as pd

# Run Stage 1 first across all ablations, then run Stage 2 individually across all TokenizerFams for the same LLM

# # STAGE 1
stage1_output_graph_path = Path("./ablate_results/samples/stage1/")
stage1_output_graph_path.mkdir(parents=True, exist_ok=True)
save_path = Path("./saves/ablations/")
save_path.mkdir(parents=True, exist_ok=True)

do_models = [
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct"
]
model_ablations = ["cadgn", "adgn", "gcn", "gat", "dec"]

import time
## Stage 1 regenerate samples
in_progress = gather_instances(
    s1config=s1c,
    arch_params=arch,
    ablations=model_ablations,
    llm_models=do_models,
    save_path=save_path,
    ip=True
)
print("Regenerating Samples...")
for mmd_int, insts in in_progress.items():
    for i, inst in enumerate(insts):
        a_id = model_ablations.index(inst.ablation)
        print(f"\t{inst.ablation}({a_id}) @ MMD {mmd_int}")
        t0 = time.perf_counter()
        inst_trainer = CADGNTrainer(
            inst.core,
            inst.families,
            profile=True
        )
        s1c.w_mmd = float(mmd_int)

        vald_d_recorded = []
        with inst.cuda_stage1() as ctx:
            # Configure properly.
            inst.core.eval()
            inst.core.configure_eval()
            for fam in inst.families:
                fam.eval()
                fam.configure_eval()

            for step_idx, sample in enumerate(complete_val):
                metrics, pred_embeds_per_fam, pred_edge_logits_per_fam = inst_trainer._stage1_step(
                    sample=sample,
                    config=s1c,
                )
                vald_d_recorded.append({
                    "sample_id": sample.sample_id,
                    "graph_id": sample.graph_id,
                    "metrics": {k: safe_convert_tensor(v) for k, v in metrics.items()},
                    "pred_embeds": {k: compact_tensor_to_b64(t) for k, t in pred_embeds_per_fam.items()},
                    "pred_logits": {k: compact_tensor_to_b64(t) for k, t in pred_edge_logits_per_fam.items()},
                    "order": [f.model_id for f in inst_trainer.families]
                })

        with open(stage1_output_graph_path / f"{a_id}_{inst.ablation}_all_{mmd_int}_s1-graph-preds.json", "w") as f:
            json.dump(vald_d_recorded, f)
        t1 = time.perf_counter()
        dt = t1 - t0
        print(f"\t\t>> Completed in {dt:.3f}s")


"""
abl_models = gather_stage1(
    s1config=s1c,
    arch_params=arch,
    ablations=model_ablations,
    llm_models=do_models,
    save_path=save_path,
    ip=False
)

stage2_output_graph_path = Path("./ablate_results/samples/stage2/")
stage2_output_graph_path.mkdir(parents=True, exist_ok=True)

for mdl in do_models:
    # Load the LLM
    with LLMWrapper(mdl) as llm:
        # Select elements to move to GPU
        for mmd_int, insts in abl_models.items():
            evals2c.w_mmd = float(mmd_int)  # this is still relevant here: does including a reconstruction penalty improve performance vs just optimizing for gate_classifier?
            for inst in insts:
                fams = inst.get_families_for_model(mdl)
                assert len(fams) == 1, f"Expected exactly 1 family for {mdl}, got {len(fams)}"
                fam: TokenizerFamily = fams[0]
                with CudaDevice(inst.core, fam):
                    safename = fam.model_id.replace("/", "__")
                    print(f"Stage 2 | abl={inst.ablation} | mmd={mmd_int} | model={mdl}")
                    print("\t>> Gathering Samples....")
                    yes_ids, no_ids = llm.get_yn_token_sets(fam.tokenizer)
                    trainer_i = CADGNTrainer(inst.core, [], profile=True, device="cuda")
                    metrics, per_sample, dict_pred_embeds, = trainer_i._eval_stage2(
                        val_dataloader=complete_val,
                        config=evals2c,
                        family=fam,
                        llmw=llm,
                        yes_ids=yes_ids,
                        no_ids=no_ids,
                        do_save=True
                    )

                    print("\t>> Saving Graph Embeds for future comparison...")
                    with open(stage2_output_graph_path / f"{inst.ablation}_{safename}_{mmd_int}_s2-graph-preds.json",
                              "w") as f:
                        # json.dump({k: safe_convert_tensor(v) for k, v in dict_pred_embeds.items()}, f)
                        json.dump([{k: safe_convert_tensor(v) for k, v in d.items()} for d in dict_pred_embeds], f)

                    print("\t>> Saving Metrics...")
                    metrics_path = Path("./ablate_results/metrics/")
                    metrics_path.mkdir(parents=True, exist_ok=True)

                    with open(metrics_path / f"{inst.ablation}_{safename}_mmd-{mmd_int}_metrics.json", "w") as f:
                        metrics_o = {k: safe_convert_tensor(v) for k, v in metrics.items()}
                        json.dump(metrics_o, f)

                    print("\t>> Saving Samples (for statistics calculations)...")
                    save_sample_results(
                        results=per_sample,
                        path=f"./ablate_results/samples/",
                        variant=f"{inst.ablation}_{safename}_mmd-{mmd_int}",
                        epoch=epochs,
                    )

                del metrics, per_sample, dict_pred_embeds, yes_ids, no_ids
                gc.collect()
                flush_gpu()
"""

