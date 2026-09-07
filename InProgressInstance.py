from CADGNTrainer import Stage1Config
from cadgn import CADGNCore, TokenizerFamily
from CudaDevice import CudaDevice
from pathlib import Path
from search import ArchParams
import torch.nn as nn
from ablations.adgn_encoder import ADGNEncoder
from ablations.gat_encoder import GATEncoder
from ablations.gcn_encoder import GCNEncoder
import torch
from typing import Union


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


def gather_instances(
        s1config: Stage1Config,
        arch_params: ArchParams,
        ablations: list[str],
        llm_models: list[str],
        save_path: Union[Path, str],
        ip=True
) -> dict[int, list["InProgressInstance"]]:
    ls: dict[int, list["InProgressInstance"]] = {0: [], 1: []}
    for a_id, abl in enumerate(ablations):
        for mmd_val in [0.0, 1.0]:
            print(f"Reloading from checkpoint for Stage 1 (ablation={abl}({a_id}), MMD Weight={mmd_val})")
            s1config.w_mmd = mmd_val
            mmd_str = f"mmd-{int(mmd_val)}"
            inst = InProgressInstance.create(
                arch=arch_params,
                abl=abl,
                llm_models=llm_models
            )

            # Load core parameters from disk
            inst.core.load_state_dict(torch.load(save_path / abl / f"CADGNCore_{mmd_str}_weights.pt", map_location="cpu", weights_only=True))

            for fam in inst.families:
                safename = fam.model_id.replace("/", "__")
                if ip:
                    fam.load_state_dict(torch.load(save_path / abl / "stage1_backup_fams" / f"TokFam_inprogress_{safename}_{mmd_str}_weights.pt", map_location="cpu", weights_only=True))
                else:
                    fam.load_state_dict(torch.load(
                        save_path / abl  / f"TokFam_{safename}_{mmd_str}_weights.pt",
                        map_location="cpu", weights_only=True))
            ls[int(mmd_val)].append(inst)
            print("\t>> Complete!")
    return ls