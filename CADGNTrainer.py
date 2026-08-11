from pathlib import Path
from typing import Iterable, Optional, TypedDict
from inspect import isdatadescriptor

#from cadgn.graph_utils import GraphBatch, _RequiredBatchFields
from dataclasses import dataclass
from CADGNCore import CADGNCore
from TokenizerFamily import TokenizerFamily
from losses import _generate_candidate_pairs, reconstruction_loss, generalized_mmd_loss
from ds.cladder import CLadderSample, CLadderDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn


@dataclass
class Stage1Config:
    """
    All hyperparameters for Stage 1 training. (grid search over these and the model hyperparameters as well)
    Preferred init via dict:
        config = Stage1Config(**param_grid_entry)
    """
    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Training Loop
    epochs: int = 100
    grad_clip: float = 1.0  # 0.0 disables clipping

    # Top-Level Loss Weights
    w_nodes: float = 1.0   # node reconstruction
    w_edges: float = 1.0   # edge reconstruction
    w_norm : float = 0.01  # embedding norm preservation
    w_mmd  : float = 1.0   # cross-family MMD alignment for EncoderHeads & pre/post projectors

    # Node Reconstruction Sub-Weights
    w_mse  : float = 1.0
    w_cosine:float = 1.0

    # Norm Loss Sub-Weights
    norm_w_preserve: float = 1.0
    norm_w_floor   : float = 1.0
    norm_epsilon   : float = 1e-8

    # MMD
    mmd_kernel: str = 'imq'
    mmd_beta: float= 0.5

    # Edge Loss
    auto_pos_weight: bool = True
    reduction: str = 'mean'


@dataclass
class Stage2Config:
    """
    All Hyperparameters for Stage 2 training
    """
    # Optimizer
    lr: float = 1e-4
    weight_decay: float = 1e-5

    # Training Loop
    epochs: int = 100
    grad_clip: float = 1.0  # 0.0 disables clipping

    reduction: str = 'mean'


class CADGNTrainer:
    """
    Orchestrates Stage 1 and Stage 2 training for the CA-DGN Framework
    Not an nn.Module.
    """
    def __init__(
            self,
            core: CADGNCore,
            families: Iterable[TokenizerFamily],
            device: str | torch.device = "cuda",
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.core: CADGNCore = core.to(device)
        self.families = list(families)
        for fam in self.families:
            fam.to(device)

    # Persistence
    def _save_checkpoint(
            self,
            checkpoint_dir: str | Path,
            epoch: int
    ) -> None:
        f"""
        Save core and all families under a shared epoch directory
        Layout:
            {checkpoint_dir}/epoch_{epoch:04d}/
                cadgn_core/
                {model_id}/ (one per family)
        :param checkpoint_dir: directory to checkpoint to. 
        :param epoch: the current epoch
        :return: None
        """
        root = Path(checkpoint_dir) / f"epoch_{epoch:04d}"

        self.core.save(path=root / "cadgn_core", model_id=f"{self.core.encoder.conv.__name__[:10]}-{self.core.encoder.hidden_dim}")

        for fam in self.families:
            safe_name = fam.model_id.replace("/", "-")
            fam.save(root / safe_name, model_id=fam.model_id)

    # Optimizer Construction
    def _build_stage1_optimizer(
            self,
            config: Stage1Config
    ) -> torch.optim.AdamW:
        """
        Joint optimizer over core + all TokenizerFamily Stage 1 parameters.
        Must be called after `configure_stage1()` on all components.
        :param config: The Config of Hyperparameters to use to train Stage 1.
        :return: AdamW Optimizer
        """
        params = list(self.core.trainable_parameters) + [
            p for fam in self.families for p in fam.trainable_parameters
        ]
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)

    def _build_stage2_optimizer(
            self,
            family: TokenizerFamily,
            config: Stage2Config
    ):
        """
        Optimizer scoped to one family's Stage 2 parameters only.
        Must be called after `family.configure_stage2()` and `core.configure_stage2()`
        :param config: The Stage 2 Config to use
        :return: AdamW Optimizer
        """
        return torch.optim.AdamW(
            list(family.trainable_parameters),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def _stage1_step(
            self,
            sample: CLadderSample,
            config: Stage1Config
    ) -> dict[str, torch.Tensor]:
        """
        Single Forward Pass for Stage 1. Runs the full pipeline once per family. Candidate paris are generated once and shared (graph topology identical)
        :param batch:
        :param config:
        :return: Flat metrics dict (only 'loss' is gradient-attached)
        """
        edge_index = sample.data.edge_index.to(self.device)
        node_texts = sample.node_names


        # Generate candidate pairs
        candidate_pairs = _generate_candidate_pairs(
            num_nodes=len(node_texts),
            device=self.device,
        )

        H_dict: dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=self.device)
        metrics: dict[str, torch.Tensor] = {}

        for family in self.families:
            name = family.model_id

            # Tokenize
            tokens = family.tokenizer(
                node_texts,
                padding=True,
                truncation=True,
                max_length=family.max_seq_len,
                return_tensors="pt",
            ).to(self.device)

            # Embeddings
            with torch.no_grad():
                embeds = family.embed_layer(tokens.input_ids)
            # (num_nodes, seq_len, llm_dim)

            # Encoder Head -> H
            H = family.encoder_head(embeds, tokens.attention_mask)
            H_dict[name] = H

            # CADGN-Encoder -> Z
            Z = self.core.encoder(H, edge_index)

            # CADGN_Decoder -> Z_dec
            Z_dec = self.core.decoder(Z)

            # DecoderHead -> pred_embeds
            pred_embeds = family.decoder_head(Z_dec, tokens.attention_mask)

            # Graph Builder -> edge_logits
            edge_logits = self.core.graph_builder(Z, candidate_pairs)

            fam_losses = reconstruction_loss(
                pred_embeds = pred_embeds,
                true_node_embeds = embeds,
                node_attention_mask = tokens.attention_mask,
                edge_logits=edge_logits,
                candidate_pairs=candidate_pairs,
                true_edge_index=edge_index,
                Z=Z,
                H=H,

                w_nodes=config.w_nodes,
                w_edges=config.w_edges,
                w_norm=config.w_norm,
                nodes_w_mse=config.w_mse,
                nodes_w_cosine=config.w_cosine,
                auto_pos_weight=config.auto_pos_weight,
                norm_w_preserve=config.norm_w_preserve,
                norm_w_floor=config.norm_w_floor,
                reduction='mean'
            )

            total_loss += fam_losses['loss']
            metrics[f"{name}/L_nodes"] = fam_losses['L_nodes']
            metrics[f"{name}/L_edges"] = fam_losses['L_edges']
            metrics[f"{name}/L_norm"] = fam_losses['L_norm']

        # EOL

        # MMD loss across all k-family H tensors (must be done at this scope)
        L_mmd = generalized_mmd_loss(
            H_dict=H_dict,
            kernel=config.mmd_kernel,
            beta=config.mmd_beta,
        )

        total_loss = total_loss + config.w_mmd * L_mmd

        return {
            "loss": total_loss,
            "L_mmd": L_mmd.detach(),
            **metrics
        }

    def train_stage1(
            self,
            #*,
            config: Stage1Config,
            train_dataloader: DataLoader[CLadderSample],
            val_dataloader: Optional[DataLoader[CLadderSample]]=None,
            checkpoint_dir: Optional[str | Path] = None,
            checkpoint_every: int = 10,
    ) -> dict[str, list[float]]:
        """
        Trains Stage 1 across all TokenizerFamily objects.
        Configures all modules for Stage 1, builds a joint optimizer, and runs the full training loop.
        :param config: Stage1Config; all training hyperparameters
        :param train_dataloader: Yields GraphBatch dicts
        :param val_dataloader: Optional; Validation metrics are computed after each epoch and logged under 'val/' keys.
        :param checkpoint_dir: Optional; Saves core + all families to disk every checkpoint_every epochs.
        :param checkpoint_every: Epoch interval for checkpointing (default 10)
        :return: history; metric name -> list of per-epoch averages.
        """
        self.core.configure_stage1()
        for fam in self.families:
            fam.configure_stage1()

        optimizer = self._build_stage1_optimizer(config=config)
        history: dict[str, list[float]] = {}

        for epoch in range(config.epochs):
            # Train
            self.core.train()
            for fam in self.families:
                fam.train()

            epoch_metrics: dict[str, list[float]] = {}

            for batch in train_dataloader:
                optimizer.zero_grad()

                step = self._stage1_step(sample=batch, config=config)
                step["loss"].backward()

                if config.grad_clip > 0.0:
                    nn.utils.clip_grad_norm_(
                        [
                            p for src in [self.core, *self.families]
                            for p in src.parameters()
                            if p.requires_grad
                        ],
                        config.grad_clip
                    )
                optimizer.step()

                for k,v in step.items():
                    epoch_metrics.setdefault(k, []).append(
                        v.item() if isinstance(v, torch.Tensor) else v
                    )

            # Epoch averages over all batches
            epoch_avg = {
                k: sum(v) / len(v)
                for k, v in epoch_metrics.items()
            }
            # Validate
            if val_dataloader is not None:
                val_avg = self._eval_stage1(val_dataloader=val_dataloader, config=config)
                for k,v in val_avg.items():
                    epoch_avg[f"val/{k}"] = v

            for k,v in epoch_avg.items():
                history.setdefault(k, []).append(v)

            # Checkpoint
            if (
                checkpoint_dir is not None
                and (epoch+1) % checkpoint_every == 0
            ):
                self._save_checkpoint(checkpoint_dir=checkpoint_dir, epoch=epoch+1)

        return history

    @torch.no_grad()
    def _eval_stage1(
            self,
            val_dataloader: DataLoader[CLadderSample],
            config: Stage1Config,
    ) -> dict[str, float]:
        """Validation pass, no gradients. Returns per-metric batch averages."""
        self.core.eval()
        for fam in self.families:
            fam.eval()

        all_metrics: dict[str, list[float]] = {}

        for batch in val_dataloader:
            step = self._stage1_step(batch=batch, config=config)
            for k,v in step.items():
                all_metrics.setdefault(k, []).append(
                    v.item() if isinstance(v, torch.Tensor) else v
                )
        return {k: sum(v) / len(v) for k, v in all_metrics.items()}

