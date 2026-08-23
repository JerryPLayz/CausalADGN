import gc

import env
import torch
import torch.nn as nn
from CADGNTrainer import CADGNTrainer, Stage1Config, Stage2Config

from models_config import models
from ds.cladder import CLadderSample, CLadderDataset, load_cladder_v1_5, CLadderLoaderConfig
from cadgn import CADGNCore, TokenizerFamily, flush_gpu, save_history, BaseEncoder, LLMWrapper, eval_and_save, get_device_of, save_sample_results
from search import ArchParams
import gc

from typing import Union, Any
from pathlib import Path
import json
from collections import defaultdict

# Unfortunately, the current implementation precludes saving modified versions of the various modules (config is static)

flush_gpu()
gc.collect()

epochs = 40

do_models = [
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct"]

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

complete_val = vald.as_dataloader()
prompt_samples: list[CLadderSample] = CLadderDataset(org_ds).stratified_sample(50)

import numpy as np
import pandas as pd
from ablations.adgn_encoder import ADGNEncoder
from ablations.gat_encoder import GATEncoder
from ablations.gcn_encoder import GCNEncoder

model_ablations = ["cadgn", "adgn", "gcn", "gat", "dec"]


class InProgressInstance:
    def __init__(self, core: CADGNCore, families: list[TokenizerFamily], ablation: str):
        self.core: CADGNCore = core
        self.families: list[TokenizerFamily] = families
        self.ablation = ablation

    def get_families_for_model(self, model_id) -> list[TokenizerFamily]:
        return [f for f in self.families if f.model_id == model_id]

    def cuda_stage1(self, device: str = "cuda"):
        return CudaDevice(self.core, *self.families, device=device)

    def cuda_stage2(self, family: TokenizerFamily, device: str = "cuda"):
        return CudaDevice(self.core, family, device=device)

    @classmethod
    def create(cls, arch: ArchParams, abl: str, llm_models):
        families = [
            TokenizerFamily.from_pretrained(
                model_id=model_id,
                max_seq_len=128,
                device="cpu",
                **arch.family_kwargs()
            )
            for model_id in llm_models
        ]

        core = CADGNCore(**arch.core_kwargs())
        if abl == "adgn":
            core.encoder = ADGNEncoder(
                hidden_dim=arch.ca_dgn_dim,
                max_layers=arch.max_layers,
                dropout=arch.encoder_dropout,
                num_iters=arch.num_iters,
                epsilon=arch.epsilon,
                base_gamma=arch.base_gamma,
            )
        elif abl == "gat":
            core.encoder = GATEncoder(
                hidden_dim=arch.ca_dgn_dim,
                max_layers=arch.max_layers,
                dropout=arch.encoder_dropout,
                num_heads=4
            )

        elif abl == "gcn":
            core.encoder = GCNEncoder(
                hidden_dim=arch.ca_dgn_dim,
                max_layers=arch.max_layers,
                dropout=arch.encoder_dropout,
            )

        elif abl == "dec":
            core.decoder = nn.Identity()

        return cls(core, families, abl)


class CudaDevice:
    """
    Context manager that moves nn.Module(s) to a target device on entry, and restores them to their original device on exit.
    """
    def __init__(self, *modules: nn.Module, device: Union[str, torch.device] = "cuda"):
        self.modules = list(modules)
        self.device = device
        self._original_devices: list[torch.device] = []

    def __enter__(self) -> "CudaDevice":
        for module in self.modules:
            self._original_devices.append(get_device_of(module))
            module.to(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for module, original in zip(self.modules, self._original_devices):
            module.to(original)
        return False

# Run Stage 1 first across all ablations, then run Stage 2 individually across all TokenizerFams for the same LLM


def safe_convert_tensor(tensor: torch.Tensor) -> Any:
    if isinstance(tensor, torch.Tensor):
        if tensor.numel() == 1:
            return tensor.item()
        else:
            return tensor.tolist()
    else:
        return tensor


in_progress: dict[int, list["InProgressInstance"]] = {0: [], 1: []}
# # STAGE 1
stage1_output_graph_path = Path("./ablate_results/samples/stage1/")
stage1_output_graph_path.mkdir(parents=True, exist_ok=True)
save_path = Path("./saves/ablations/")
save_path.mkdir(parents=True, exist_ok=True)


def do_stage1():
    for a_id, abl in enumerate(model_ablations):
        backup_dir = save_path / abl / "stage1_backup_fams"
        backup_dir.mkdir(parents=True, exist_ok=True)

        for mmd_val in [0.0, 1.0]:
            print(f"Starting Training for Stage 1 (ablation={abl}({a_id}), MMD Weight={mmd_val})")
            s1c.w_mmd = mmd_val
            mmd_str = f"mmd-{int(mmd_val)}"
            inst = InProgressInstance.create(
                    arch=arch,
                    abl=abl,
                    llm_models=do_models
                )
            # Train Stage 1 for all models (including across w_mmd=0/1)
            with inst.cuda_stage1():
                inst_trainer = CADGNTrainer(
                    inst.core,
                    inst.families,
                    profile=True
                )
                history = inst_trainer.train_stage1(
                    config=s1c,
                    train_dataloader=train_d,
                    val_dataloader=val_d,
                )

                save_history(history, f"./ablate_results/stage1_history/{a_id}_{abl}_stage1_history_{mmd_str}.csv")
                # Evaluate over the full validation set and save base loss values (for the graphs, point of comparison)
                inst.core.eval()
                inst.core.configure_eval()
                vald_d_recorded = []
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
                    "metrics": {k: safe_convert_tensor(v) for k, v in metrics.items()},
                    "pred_embeds": {k: t.tolist() for k, t in pred_embeds_per_fam.items()},
                    "pred_logits": {k: t.tolist() for k, t in pred_edge_logits_per_fam.items()},
                    "order": [f.model_id for f in inst_trainer.families]
                })

            with open(stage1_output_graph_path / f"{a_id}_{abl}_all_{mmd_str}_s1-graph-preds.json", "w") as f:
                json.dump(vald_d_recorded, f)
            in_progress[int(mmd_val)].append(inst)
            # Save the Core (as its frozen in Stage 2, this is it)
            (save_path / inst.ablation).mkdir(parents=True, exist_ok=True)
            inst.core.save(
                path = save_path / inst.ablation,
                model_id=mmd_str
            )

            for fam in inst.families:
                safename = fam.model_id.replace("/", "__")
                fam.save(
                    path= backup_dir,
                    model_id=f"inprogress_{safename}_{mmd_str}",
                )
            del inst_trainer
            del vald_d_recorded
        # EOL


def gather_stage1() -> dict[int, list["InProgressInstance"]]:
    ls: dict[int, list["InProgressInstance"]] = {0: [], 1: []}
    for a_id, abl in enumerate(model_ablations):
        for mmd_val in [0.0, 1.0]:
            print(f"Reloading from checkpoint for Stage 1 (ablation={abl}({a_id}), MMD Weight={mmd_val})")
            s1c.w_mmd = mmd_val
            mmd_str = f"mmd-{int(mmd_val)}"
            inst = InProgressInstance.create(
                arch=arch,
                abl=abl,
                llm_models=do_models
            )

            # Load core parameters from disk
            inst.core.load_state_dict(torch.load(save_path / abl / f"CADGNCore_{mmd_str}_weights.pt", map_location="cpu", weights_only=True))

            for fam in inst.families:
                safename = fam.model_id.replace("/", "__")
                fam.load_state_dict(torch.load(save_path / abl / "stage1_backup_fams" / f"TokFam_inprogress_{safename}_{mmd_str}_weights.pt", map_location="cpu", weights_only=True))

            ls[int(mmd_val)].append(inst)
            print("\t>> Complete!")
    return ls


in_progress = gather_stage1()
#do_stage1()


# # Stage 2 (careful memory management is painful to work with...)
stage2_output_graph_path = Path("./ablate_results/samples/stage2/")
stage2_output_graph_path.mkdir(parents=True, exist_ok=True)

for mdl in do_models:
    # Load the LLM
    with LLMWrapper(mdl) as llm:
        # Select elements to move to GPU
        for mmd_int, insts in in_progress.items():
            s2c.w_mmd = float(mmd_int)  # this is still relevant here: does including a reconstruction penalty improve performance vs just optimizing for gate_classifier?
            for inst in insts:
                fams = inst.get_families_for_model(mdl)
                assert len(fams) == 1, f"Expected exactly 1 family for {mdl}, got {len(fams)}"
                fam: TokenizerFamily = fams[0]
                with CudaDevice(inst.core, fam):
                    safename = fam.model_id.replace("/", "__")
                    print(f"Stage 2 | abl={inst.ablation} | mmd={mmd_int} | model={mdl}")
                    stage2_trainer_i = CADGNTrainer(inst.core, [], profile=True, device="cuda")
                    history_i = stage2_trainer_i.train_stage2(
                        config=s2c,
                        family=fam,
                        train_dataloader=train_d,
                        val_dataloader=val_d,
                        llmw=llm
                    )

                    print("\t>> Saving History")

                    save_history(
                        history=history_i,
                        path=f"./ablate_results/history/{inst.ablation}_{safename}_mmd-{mmd_int}_history.csv"
                    )

                    print("\t>> Gathering Samples....")
                    yes_ids, no_ids = llm.get_yn_token_sets(fam.tokenizer)
                    metrics, per_sample, dict_pred_embeds, = stage2_trainer_i._eval_stage2(
                        val_dataloader=val_d,
                        config=s2c,
                        family=fam,
                        llmw=llm,
                        yes_ids=yes_ids,
                        no_ids=no_ids,
                        do_save = True
                    )

                    print("\t>> Saving Graph Embeds for future comparison...")
                    with open(stage2_output_graph_path / f"{inst.ablation}_{safename}_{mmd_int}_s2-graph-preds.json"):
                        json.dump(dict_pred_embeds, f)


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

                    # this is an expensive process, only do it once for each of the mmd_val variants (max x4)
                    if inst.ablation == "cadgn" or inst.ablation == "adgn":
                        print("\t>> Handling Qualitative Gather...")

                        o_texts = []
                        output_path = Path("./ablate_results/output")
                        output_path.mkdir(parents=True, exist_ok=True)

                        for sample_idx, sample in enumerate(prompt_samples):

                            input_embeds = llm.embed_prompt(
                                prompt=sample.prompt,
                                tokenizer=fam.tokenizer,
                                embed_layer=fam.embed_layer,
                                device=llm.device
                            )

                            _, _, _, _, _, graph_embeds = stage2_trainer_i.generate_graph_embeds_s2(
                                sample=sample,
                                family=fam,
                                llm=llm,
                                detached=True
                            )

                            input_embeds, attention_mask = llm.assemble_inputs(
                                graph_embeds=graph_embeds,
                                prompt_embeds=input_embeds,

                            )

                            token_ids = llm.generate_text(
                                input_embeds=input_embeds,
                                attention_mask=attention_mask,
                                tokenizer=fam.tokenizer,
                            )

                            text = fam.tokenizer.decode(token_ids, skip_special_tokens=True)
                            o_texts.append({
                                "id": sample.sample_id,
                                "rung": sample.rung,
                                "llm_output": text,
                                "prompt": sample.prompt,
                                "query_type": sample.query_type,
                                "reasoning": sample.reasoning,
                                "formal_form": sample.formal_form
                            })

                        o_texts = pd.DataFrame(o_texts)
                        o_texts.to_csv(f"./ablate_results/output/{inst.ablation}_{safename}_mmd-{mmd_int}_qual.csv")
                        print("\t>> Saved Qualitative Data!")
                    # EOIF
                    # Save to disk
                    (save_path / inst.ablation).mkdir(parents=True, exist_ok=True)
                    fam.save(
                        path=save_path / inst.ablation,
                        model_id=f"{safename}_mmd-{mmd_int}"
                    )
