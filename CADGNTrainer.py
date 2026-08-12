from pathlib import Path
from typing import Iterable, Optional, TypedDict, Any
from inspect import isdatadescriptor

#from cadgn.graph_utils import GraphBatch, _RequiredBatchFields
from dataclasses import dataclass
from CADGNCore import CADGNCore
from TokenizerFamily import TokenizerFamily
from losses import _generate_candidate_pairs, reconstruction_loss, generalized_mmd_loss
from ds.cladder import CLadderSample, CLadderDataset
from torch.utils.data import DataLoader
from graph_visualizer import visualize_graph_diff

import torch
import torch.nn as nn
from cadgn import Profiler
import matplotlib.pyplot as plt

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

    # Dataset sampling & Gradient Accumulation
    max_steps_per_epoch: Optional[int] = None  # full dataset per epoch; limit to cutoff samples after a point (due to CyclicSubsetSampler, we guarantee all samples will be seen across epochs)
    grad_accum_steps: int = 1  # 1 = batch-size of 1, (simulate larger batch-sizes for efficiency by calling `optimizer.step` on many independent samples at once.)
    max_steps_per_epoch_val: Optional[int] = None  # validation (same as above, but for validation set)


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
            profile: bool = False,
            profile_sync: bool = True,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.core: CADGNCore = core.to(device)
        self.families = list(families)
        self.profiler = Profiler(
            sync_cuda=profile_sync,
            enabled=profile
        )
        for fam in self.families:
            fam.to(device)

    # Persistence
    def _save_checkpoint(
            self,
            checkpoint_dir: str | Path,
            epoch: int
    ) -> None:
        """
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

    def _stage1_p1(
            self,
            family: TokenizerFamily,
            node_texts: list[str],
            edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Part 1 of Stage 1: Tokenize Node Texts -> Embed Node Texts -> Encode/Project -> H -> CADGNEncoder -> Z
        :param family: the TokenizerFamily to run this computation for.
        :param node_texts: the original node texts in a list/iterable
        :param edge_index: edge logts in standard networkx format. [[src], [dst]]
        :return: (H, Z, embeds, attention_mask)
        """
        # Tokenize
        with self.profiler.section(f"S1 // 2-{family.sh_id}-tokenizer"):
            tokens = family.tokenizer(
                node_texts,
                padding=True,
                truncation=True,
                max_length=family.max_seq_len,
                return_tensors="pt",
            ).to(self.device)

        # print(f"Token Shapes: {tokens.input_ids.shape}")

        # Embeddings
        with self.profiler.section(f"S1 // 3-{family.sh_id}-embed_layer"):
            with torch.no_grad():
                embeds = family.embed_layer(tokens.input_ids)
        # (num_nodes, seq_len, llm_dim)

        # print(f"Pre-Encoder Head: {embeds.shape}")
        # Encoder Head -> H
        with self.profiler.section(f"S1 // 4-{family.sh_id}-enc_head"):
            H = family.encoder_head(embeds, tokens.attention_mask)
        # print(f"Pre-CADGN Enc: {H.shape}")

        # CADGN-Encoder -> Z
        with self.profiler.section(f"S1 // 5-core-cadgn_encoder"):
            Z = self.core.encoder(H, edge_index)
        return H, Z, embeds, tokens.attention_mask

    def _stage1_p2(
            self,
            family: TokenizerFamily,
            Z: torch.Tensor,
            candidate_pairs: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        CADGNDeocder -> Z_dec -> DecoderHead -> pred_embeds

        GraphBuilder -> edge_logits
        :param family: A TokenizerFamily object
        :param Z: CADGN-encoded tensor
        :param candidate_pairs: candidate pairs for graph reconstruction, generated by cadgn.graph_utils._generate_candidate_pairs
        :param attention_mask: The token mask to appropriately decode the node tokens from the embed.
        :return: (pred_embeds, edge_logits)
        """
        with self.profiler.section(f"S1 // 6-core-cadgn_decoder"):
            Z_dec = self.core.decoder(Z)

        # DecoderHead -> pred_embeds
        with self.profiler.section(f"S1 // 7-{family.sh_id}-dec_head"):
            pred_embeds = family.decoder_head(Z_dec, attention_mask)

        # Graph Builder -> edge_logits
        with self.profiler.section(f"S1 // 8-{family.sh_id}-gbuild"):
            edge_logits = self.core.graph_builder(Z, candidate_pairs)
        return pred_embeds, edge_logits

    def _stage1_step(
            self,
            sample: CLadderSample,
            config: Stage1Config
    ) -> tuple[
        dict[str, torch.Tensor],  # metrics (loss is gradient attached)
        dict[str, torch.Tensor],  # pred_embeds per family (detached)
        dict[str, torch.Tensor],  # pred_edge_logits per family (detached)
    ]:
        """
        Single Forward Pass for Stage 1. Runs the full pipeline once per family. Candidate paris are generated once and shared (graph topology identical)
        :param sample: an individual graph to process in CLadderSample format.
        :param config:
        :return: Flat metrics dict (only 'loss' is gradient-attached)
        """
        edge_index = sample.data.edge_index.to(self.device)
        node_texts = sample.node_names

        # Generate candidate pairs
        with self.profiler.section("S1 // 1-candidate_pairs"):
            candidate_pairs = _generate_candidate_pairs(
                num_nodes=len(node_texts),
                device=self.device,
            )

        H_dict: dict[str, torch.Tensor] = {}
        pred_embeds_dict: dict[str, torch.Tensor] = {}
        pred_edge_logits_dict: dict[str, torch.Tensor] = {}
        total_loss = torch.tensor(0.0, device=self.device)
        metrics: dict[str, torch.Tensor] = {}
        #print(f"[ID={sample.sample_id:06} : N={len(node_texts):06}]")
        for family in self.families:
            name = family.model_id
            H, Z, embeds, attention_mask = self._stage1_p1(
                family, node_texts, edge_index
            )
            H_dict[name] = H

            pred_embeds, pred_edge_logits = self._stage1_p2(
                family, Z, candidate_pairs, attention_mask
            )

            pred_embeds_dict[name] = pred_embeds.detach()
            pred_edge_logits_dict[name] = pred_edge_logits.detach()

            with self.profiler.section(f"S1 // 9-{family.sh_id}-reconloss"):
                fam_losses = reconstruction_loss(
                    pred_node_embeds = pred_embeds,
                    true_node_embeds = embeds,
                    node_attention_mask = attention_mask,
                    edge_logits=pred_edge_logits,
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
            metrics[f"{name}/L_nodes"] = fam_losses['L_nodes'].detach()
            metrics[f"{name}/L_edges"] = fam_losses['L_edges'].detach()
            metrics[f"{name}/L_norm"] = fam_losses['L_norm'].detach()

        # EOL

        # MMD loss across all k-family H tensors (must be done at this scope)
        with self.profiler.section(f"S1 // 10-mmd_loss"):
            L_mmd = generalized_mmd_loss(
                Z_dict=H_dict,
                kernel=config.mmd_kernel,
                beta=config.mmd_beta,
            )

        total_loss = total_loss + config.w_mmd * L_mmd

        return (
            {
                "loss": total_loss,
                "L_mmd": L_mmd.detach(),
                **metrics
            },
            pred_embeds_dict,
            pred_edge_logits_dict
        )

    def train_stage1(
            self,
            # *,
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
        print("Training Stage 1 for:")
        print(f"\tCore: {self.core}")
        print(f"\tFamilies: {[f.model_id for f in self.families]}")

        for epoch in range(config.epochs):
            # with self.profiler.section(f"S1 // 0-Epoch-Summary"):
            # Train
            self.core.train()
            for fam in self.families:
                fam.train()

            epoch_metrics: dict[str, list[float]] = {}
            optimizer.zero_grad()
            with self.profiler.section("S1 // Epoch"):
                for step_idx, sample in enumerate(train_dataloader):
                    with self.profiler.section(f"S1 // 00 - Step"):
                        metrics, _, _ = self._stage1_step(sample=sample, config=config)
                    loss = metrics["loss"] / config.grad_accum_steps
                    loss.backward()

                    is_accum_step = (step_idx + 1) % config.grad_accum_steps == 0
                    is_last_step = (
                        config.max_steps_per_epoch is not None
                        and step_idx + 1 >= config.max_steps_per_epoch
                    )
                    # simulate larger batch sizes to promote greater stability during training
                    if is_accum_step or is_last_step:
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
                        optimizer.zero_grad()
                    for k,v in metrics.items():
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
                with self.profiler.section("S1 // Epoch (Vald)"):
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

            train_loss = epoch_avg.get("loss", float("nan"))
            val_loss = epoch_avg.get("val/loss", float("nan"))
            print(
                f"Epoch {epoch:>4d} | "
                f"Train loss: {train_loss:.5f} | "
                f"Vald  loss: {val_loss:.5f}",
                end=""
            )
            if self.profiler.enabled:
                elapsed_t = self.profiler.get_latest_n('S1 // Epoch')
                elapsed_v = self.profiler.get_latest_n('S1 // Epoch (Vald)')
                tot_elapsed = elapsed_t + elapsed_v
                print(
                    f" | Elapsed: {tot_elapsed:.3f}s (t:{elapsed_t:.2f} + v:{elapsed_v:.2f}) "
                )

            else:
                print(f"")
        # end of epochs...
        if self.profiler.enabled:
            print(f"Profiler Summary:")
            print(self.profiler.summary(sort_by="total"))

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
            metrics, _, _ = self._stage1_step(sample=batch, config=config)
            for k,v in metrics.items():
                all_metrics.setdefault(k, []).append(
                    v.item() if isinstance(v, torch.Tensor) else v
                )
        return {k: sum(v) / len(v) for k, v in all_metrics.items()}

    def _inference_stage1(
            self,
            sample:CLadderSample,
            config,
            family: TokenizerFamily,
            figure: bool = True
    ) -> dict[str, Any]:
        # todo: redo: this isn't correct.
        metrics, pred_embeds, edge_logits = self._stage1_step(sample=sample, config=config)
        model_id = family.model_id
        fig = visualize_graph_diff(
            sample=sample,
            edge_logits=edge_logits[model_id],
            candidate_pairs=_generate_candidate_pairs(
                num_nodes=len(sample.node_names),
                device=config.device,
            )
        )
        return {"figure":fig}
