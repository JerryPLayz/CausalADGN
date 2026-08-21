from pathlib import Path
from typing import Iterable, Optional, Callable, Literal
from contextlib import nullcontext

#from cadgn.graph_utils import GraphBatch, _RequiredBatchFields
from dataclasses import dataclass
from cadgn.CADGNCore import CADGNCore
from cadgn.TokenizerFamily import TokenizerFamily, DTYPE_MAP
from losses import _generate_candidate_pairs, reconstruction_loss, generalized_mmd_loss
from ds.cladder import CLadderSample
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadgn import Profiler, LLMWrapper, LLMOutputs, GateClassifier, Stage2Intermediates
from torchmetrics.classification import MulticlassF1Score, BinaryF1Score
from cadgn import BaselineSampleResult, EarlyStopping


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
    early_stopping_patience: int = 10

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
    early_stopping_patience: int = 10

    # Dataset sampling & Gradient Accumulation
    grad_accum_steps: int = 1  # 1 = batch-size of 1, (simulate larger batch-sizes for efficiency by calling `optimizer.step` on many independent samples at once.)
    max_steps_per_epoch: Optional[int] = None  # full dataset per epoch; limit to cutoff samples after a point (due to CyclicSubsetSampler, we guarantee all samples will be seen across epochs)
    max_steps_per_epoch_val: Optional[int] = None  # validation (same as above, but for validation set)

    # MMD
    mmd_kernel: Literal["imq", "rbf"] = 'imq'
    mmd_beta: float = 0.5

    # Graph Injection
    graph_first: bool = True

    # losses
    w_mmd: float = 1.0
    w_gate: float = 1.0


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
        Single Forward Pass for Stage 1. Runs the full pipeline once per family. Candidate pairs are generated once and shared (graph topology identical)
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
            pruning_callback: Optional[Callable[[int, float], None]] = None,
    ) -> dict[str, list[float]]:
        """
        Trains Stage 1 across all TokenizerFamily objects.
        Configures all modules for Stage 1, builds a joint optimizer, and runs the full training loop.
        :param config: Stage1Config; all training hyperparameters
        :param train_dataloader: Yields GraphBatch dicts
        :param val_dataloader: Optional; Validation metrics are computed after each epoch and logged under 'val/' keys.
        :param checkpoint_dir: Optional; Saves core + all families to disk every checkpoint_every epochs.
        :param checkpoint_every: Epoch interval for checkpointing (default 10)
        :param pruning_callback: Optional callback for pruning during parameter search (default None)
        :return: history; metric name -> list of per-epoch averages.
        """
        self.core.configure_stage1()
        for fam in self.families:
            fam.configure_stage1()

        if pruning_callback is not None and val_dataloader is None:
            raise ValueError("pruning_callback requires val_dataloader to be provided: pruning decisions are based on val/loss.")

        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=1e-4,
            mode="min",
        )
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

            if pruning_callback is not None:
                pruning_callback(epoch, val_loss)

            if val_dataloader is not None and early_stopping.step(val_avg['loss'], epoch):
                print(f"\t>> Early Stopping at epoch {epoch:>4d} | best: {early_stopping.best_epoch:>4d}")
                break

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
        self.core.configure_eval()
        for fam in self.families:
            fam.eval()
            fam.configure_eval()

        all_metrics: dict[str, list[float]] = {}

        for step_idx, sample in enumerate(val_dataloader):
            metrics, _, _ = self._stage1_step(sample=sample, config=config)
            for k,v in metrics.items():
                all_metrics.setdefault(k, []).append(
                    v.item() if isinstance(v, torch.Tensor) else v
                )
        return {k: sum(v) / len(v) for k, v in all_metrics.items()}

    def _stage2_step(
            self,
            sample: CLadderSample,
            config: Stage2Config,
            family: TokenizerFamily,
            llm: LLMWrapper,
            inference: bool = False,
    ) -> tuple[
        dict[str, torch.Tensor],  # metrics (loss is gradient attached)
        torch.Tensor,  # pred_embeds (detached)
        torch.Tensor,  # pred_edge_logits (detached)
        torch.Tensor,  # gate_classifier logits (caller must compute L_gate)
        LLMOutputs,
        Stage2Intermediates,
    ]:
        """
        Single forward pass for Stage 2. Runs the full pipeline for one family.
        Gradient Flow:
            L_mmd -> Z_approx -> Post Projector -> LLM (frozen) -> Z_intermediate -> Pre Projector -> Z
            see gate classifier logits in return for gradients for Gate Classifier (to do by caller)
        :param config:
        :param family: The LLM / TokenizerFamily to train for.
        :return: (dict: loss_dict, Tensor: pred_embeds (nodes), Tensor: pred_edge_logits (edges), Tensor: gate classifier logits, LLMOutputs, Stage2Intermediates: dataclass of intermediate (detached) tensors for evaluation)
        """
        candidate_pairs, H, Z, embeds, attention_mask, intermediate_Z = self.generate_graph_embeds_s2(
            sample=sample,
            family=family,
            llm=llm,
            detached=False
        )

        with self.profiler.section("S2 // 3-LLM"):
            prompt_embeds = llm.embed_prompt(sample.prompt, family.tokenizer, family.embed_layer, self.device)
            # (prompt_len + suffix_len, llm_dim)

            input_embeds, llm_mask = LLMWrapper.assemble_inputs(
                graph_embeds=intermediate_Z,
                prompt_embeds=prompt_embeds,
                graph_first=config.graph_first,
            )
            # input_embeds: (1, num_nodes + prompt_len, llm_dim)
            # nb: graph nodes may be first or last depending on param above.

            llm_outputs = llm.forward(
                input_embeds=input_embeds,
                attention_mask=llm_mask,
                num_nodes=len(sample.node_names),
                graph_first=config.graph_first,
                inference=inference,  # preserve computation graph for PreProjector (during .backwards())
            )
            # llm_outputs.graph_hidden: (num_nodes, llm_dim)
            # llm_outputs.last_token_hidden: (llm_dim,)
        # Post-LLM
        with self.profiler.section("S2 // 4-PostLLM"):
            # Post Projector
            Z_approx = family.post_projector(llm_outputs.graph_hidden)
            # (num_nodes, ca_dgn_dim)

            # Shared decoders
            pred_embeds, pred_edge_logits = self._stage1_p2(
                family=family,
                Z=Z_approx,
                candidate_pairs=candidate_pairs,
                attention_mask=attention_mask,
            )
            # pred_embeds: (num_nodes, ca_dgn_dim), pred_edge_logits: (num_nodes * num_nodes)

            # Gate Classifier on last token of full [graph + prompt] sequence
            gate_logits = family.gate_classifier(llm_outputs.last_token_hidden.unsqueeze(0)).squeeze(0)
            # (3,) - raw logits over {associational, interventional, counterfactual}

        with self.profiler.section("S2 // 5-Loss"):
            # Primary loss here is MMD between Z_approx and Z.
            L_mmd = generalized_mmd_loss(
                Z_dict={
                    "Z": Z.detach(),
                    "Z_approx": Z_approx,  # trained output
                },
                kernel=config.mmd_kernel,
                beta=config.mmd_beta,
            )

            L_total = config.w_mmd * L_mmd

        return (
            {
                "loss": L_total,
                "L_mmd": L_mmd.detach(),
            },
            pred_embeds.detach(),
            pred_edge_logits.detach(),
            gate_logits,
            llm_outputs,
            Stage2Intermediates.from_step(
                H = H.detach(),
                Z = Z.detach(),
                int_Z=intermediate_Z.detach(),
                Z_approx = Z_approx.detach(),
            )

        )

    def _stage2_composite_metric(self, val_avg: dict) -> float:
        """
        Weights model performance between yes/no f1 score and gate f1 macro score.
        This should provide a solid learning space.
        :param val_avg:
        :return:
        """
        yn = val_avg.get("yn_f1", 0.0)
        gate = val_avg.get("gate_f1_macro", 0.0)
        return 0.7 * yn + 0.3 * gate

    def train_stage2(
            self,
            config: Stage2Config,
            family: TokenizerFamily,
            train_dataloader: DataLoader[CLadderSample],
            val_dataloader: Optional[DataLoader[CLadderSample]],
            llmw: LLMWrapper,
            checkpoint_dir: Optional[str | Path] = None,
            checkpoint_every: int = 10,

    ):
        """
        Trains Stage 2 across one family (llm) and the core.


        :param config: Stage2Config; all training hyperparameters
        :param family: The specific TokenizerFamily to train on (i.e. has to load this Family's LLM)
        :param train_dataloader: Yeilds 'batches' to train on. (single sample batches)
        :param val_dataloader: Optional; validation metrics are computed on this data.
        :param checkpoint_dir: Optional; Saves core + families to disk every checkpoint_every epochs.
        :param checkpoint_every:Epoch interval for checkpointing (default 10)
        :param llmw: If not None, this `LLMWrapper` will be used as context instead of
        :return: history; metric name -> list of per-epoch averages
        """
        llmw.require_loaded()
        self.core.configure_stage2()
        family.configure_stage2()

        optimizer = self._build_stage2_optimizer(family=family, config=config)
        history: dict[str, list[float]] = {}
        llm_dtype = DTYPE_MAP.get(family.__config__["llm_dtype"], torch.bfloat16)

        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=1e-4,
            mode="max",
        )

        llm = llmw
        print(f"Training Stage 2 for {family.model_id}:")
        # Context-load LLM via LLMWrapper to ensure proper disposal after training.

        yes_ids, no_ids = llm.get_yn_token_sets(family.tokenizer)
        for epoch in range(config.epochs):
            self.core.train()
            family.train()


            epoch_metrics: dict[str, list[float]] = {}
            optimizer.zero_grad()

            with self.profiler.section("S2 // Epoch"):

                for step_idx, sample in enumerate(train_dataloader):
                    with self.profiler.section("S2 // 00 - Step"):
                        metrics, pred_node_embeds, pred_edge_logits, gate_cls_logits, llm_outputs, intermediates = self._stage2_step(sample=sample, config=config, family=family, llm=llm)
                    # Gate Classifier Loss add

                    metrics["L_gate"] = GateClassifier.calc_loss(
                        gate_cls_logits=gate_cls_logits,
                        rung_t=sample.rung,
                        device=self.device,
                    )
                    metrics["loss"] = metrics["loss"] + metrics["L_gate"] * config.w_gate
                    loss = metrics["loss"] / config.grad_accum_steps
                    loss.backward()

                    is_accum_step = (step_idx + 1) % config.grad_accum_steps == 0
                    is_last_step = (
                        config.max_steps_per_epoch is not None
                        and step_idx + 1 >= config.max_steps_per_epoch
                    )

                    if is_accum_step or is_last_step:
                        if config.grad_clip > 0.0:
                            nn.utils.clip_grad_norm_(
                                [
                                    p for src in [self.core, family]
                                    for p in src.parameters()
                                    if p.requires_grad
                                ],
                                config.grad_clip
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    for k, v in metrics.items():
                        epoch_metrics.setdefault(k, []).append(
                            v.item() if isinstance(v, torch.Tensor) else v
                        )
            # Post-epoch
            epoch_avg = {
                k: sum(v) / len(v)
                for k, v in epoch_metrics.items()
            }

            # Validate
            if val_dataloader is not None:
                with self.profiler.section("S2 // Epoch (Vald)"):
                    val_avg, _ = self._eval_stage2(
                        val_dataloader=val_dataloader,
                        config=config,
                        family=family,
                        llmw=llm,
                        yes_ids=yes_ids,
                        no_ids=no_ids,
                    )
                    #self.core.configure_stage2()
                    #family.configure_stage2()
                    self.core.train()
                    family.train()

                for k, v in val_avg.items():
                    epoch_avg[f"val/{k}"] = v

            for k, v in epoch_avg.items():
                history.setdefault(k, []).append(v)

            if (
                checkpoint_dir is not None
                and (epoch + 1) % checkpoint_every == 0
            ):
                root = Path(checkpoint_dir) / f"epoch_{epoch:04d}"
                self.core.save(path=root / "cadgn_core", model_id=f"{self.core.encoder.conv.__name__[:10]}-{self.core.encoder.hidden_dim}")
                safe_name = family.model_id.replace("/", "-")
                family.save(path=root / safe_name, model_id=family.model_id)

            # Print Epoch Statistics
            print(
                f"Epoch {epoch:>4d} | "
                f"Train Loss: {epoch_avg.get('loss', float('nan')):.6f}",
                end=""
            )
            if val_dataloader is not None:
                print(
                    f" | Val Loss: {epoch_avg.get('val/loss', float('nan')):.6f} | "
                    f"Accuracy: {epoch_avg.get('val/accuracy', float('nan')):.4f}",
                    end=""
                )
            if self.profiler.enabled:
                print(
                    f" | {self.profiler.get_latest_n('S2 // Epoch'):.2f}s"
                )
            else:
                print("")

            if val_dataloader is not None:
                monitor = self._stage2_composite_metric(val_avg)
                if early_stopping.step(monitor, epoch):
                    print(f"Early stopping at epoch {epoch} (best: {monitor} at {early_stopping.best_epoch})")
                    break


        # end of epochs
        if self.profiler.enabled:
            print(f"Profiler Summary:")
            print(self.profiler.summary(sort_by="total"))
        return history

    @torch.no_grad()
    def _eval_stage2(
            self,
            val_dataloader: DataLoader[CLadderSample],
            config: Stage2Config,
            family: TokenizerFamily,
            llmw: Optional[LLMWrapper],
            yes_ids: set[int],
            no_ids: set[int],
    ) -> tuple[ dict[str, float], list[BaselineSampleResult]]:
        """
        Validation pass for Stage 2.
        Computes loss metrics (same as training), plus yes/no accuracy, confidence and yn_coverage diagnostics.

        :param val_dataloader:
        :param config:
        :param family:
        :param llmw: optional - a wrapper context containing the LLM to use for inference. If not provided, will be created (and load the family.model_id LLM into memory)
        :param yes_ids: from `llm.get_yn_token_sets()`
        :param no_ids:  from `llm.get_yn_token_sets()`
        :return: (dict of per-metric averages over validation samples. ;; list of BaselineSampleResults)
        """
        self.core.eval()
        family.eval()
        self.core.configure_eval()
        family.configure_eval()

        all_metrics: dict[str, list[float]] = {}
        per_sample: list[BaselineSampleResult] = []
        llm_dtype = DTYPE_MAP.get(family.__config__["llm_dtype"], torch.bfloat16)

        ctx = (
            nullcontext(llmw)
            if llmw is not None
            else LLMWrapper(
                family.model_id,
                device=self.device,
                torch_dtype=llm_dtype,
            )
        )

        # F1 Trackers
        gate_f1 = MulticlassF1Score(
            num_classes=3,
            average="none",  # per-class F1
        ).to(device=self.device)

        gate_f1_macro = MulticlassF1Score(
            num_classes=3,
            average="macro",  # unweighted mean
        ).to(device=self.device)

        # Yes/No macro (exclude abstain samples)
        yn_f1 = BinaryF1Score().to(device=self.device)
        abstain_count = 0
        total_count = 0

        with ctx as llm:
            for step_idx, sample in enumerate(val_dataloader):
                total_count += 1
                metrics, pred_embeds, pred_edge_logits, gate_logits, llm_outputs, intermediates = self._stage2_step(
                    sample=sample,
                    config=config,
                    family=family,
                    llm=llm,
                    inference=True
                )

                # GateClassifier evaluation
                L_gate = GateClassifier.calc_loss(
                    gate_cls_logits=gate_logits,
                    rung_t=sample.rung,
                    device=self.device,
                )

                gate_pred = int(gate_logits.argmax().item())
                gate_correct = GateClassifier.is_correct(gate_logits, sample.rung)

                rung_target = torch.tensor([sample.rung], device=self.device)
                gate_pred_t = gate_logits.argmax().unsqueeze(0)

                gate_f1.update(gate_pred_t, rung_target)
                gate_f1_macro.update(gate_pred_t, rung_target)

                # Yes/No Classification (CLadder)
                # noinspection bad-argument-type
                prediction, confidence, yn_coverage = LLMWrapper.classify_yn(
                    logits=llm_outputs.next_token_logits,
                    yes_ids=yes_ids,
                    no_ids=no_ids,
                )

                if prediction != "abstain":
                    yn_pred = torch.tensor(
                        [1 if prediction == "yes" else 0], device=self.device
                    )
                    yn_target = torch.tensor(
                        [1 if sample.label == "yes" else 0], device=self.device
                    )
                    yn_f1.update(yn_pred, yn_target)
                else:
                    abstain_count += 1

                yn_correct = (
                    float(prediction == sample.label)
                    if prediction != "abstain"  # allow the model to safely abstain
                    else 0.5
                )

                per_sample.append(BaselineSampleResult(
                    sample_id=sample.sample_id,
                    rung=sample.rung,
                    label=sample.label,
                    prediction=prediction,
                    yn_correct=yn_correct,
                    confidence=confidence,
                    yn_coverage=yn_coverage,
                    gate_pred=gate_pred,
                    gate_correct=gate_correct,
                    gate_logits=gate_logits.tolist(),  # detach implicit by this operation
                ))

                # Accumulate
                metrics["L_gate"] = L_gate
                metrics["loss"] = metrics["loss"] + metrics["L_gate"] * config.w_gate
                for k, v in metrics.items():
                    all_metrics.setdefault(k, []).append(
                        v.item() if isinstance(v, torch.Tensor) else v
                    )

                all_metrics.setdefault("accuracy", []).append(yn_correct)
                all_metrics.setdefault("confidence", []).append(confidence)
                all_metrics.setdefault("yn_coverage", []).append(yn_coverage)
                all_metrics.setdefault("gate_correct", []).append(gate_correct)
                # Per-rung breakdowns to support statistics
                all_metrics.setdefault(f"accuracy_rung{sample.rung}", []).append(yn_correct)
                all_metrics.setdefault(f"yn_coverage_rung{sample.rung}", []).append(yn_coverage)
                all_metrics.setdefault(f"abstain_rung{sample.rung}", []).append(float(prediction == "abstain"))

                # Geometric Diagnostics - directly reflects whether the projectors are properly preserving the latent structure.
                norm_ratio = (intermediates.Z_approx.norm(dim=-1).mean() / intermediates.Z.norm(dim=-1).mean().clamp(min=1e-8)).item()
                cos_sim = F.cosine_similarity(intermediates.Z_approx, intermediates.Z, dim=-1).mean().item()
                all_metrics.setdefault("norm_ratio", []).append(norm_ratio)
                all_metrics.setdefault("z_cos_sim", []).append(cos_sim)

        # Compute F1 Scores
        gate_f1_per_class = gate_f1.compute()
        # (3, ) - F1 per rung: [associational, interventional, counterfactual]
        result = {k: sum(v) / len(v) for k, v in all_metrics.items()}

        rung_names = ["assoc", "interv", "counterfact"]  # 0,1,2
        for i, name in enumerate(rung_names):
            result[f"gate_f1_{name}"] = gate_f1_per_class[i].item()

        result["gate_f1_macro"] = gate_f1_macro.compute().item()
        result["yn_f1"] = yn_f1.compute().item() if total_count > abstain_count else 0.0
        result["abstain_rate"] = abstain_count / total_count if total_count > 0 else 0.0

        for rung in range(3):
            for metric in ("accuracy", "yn_coverage", "abstain"):
                key = f"{metric}_rung{rung}"
                if key not in result:
                    result[key] = float("nan")

        return result, per_sample

    def generate_graph_embeds_s2(
            self,
            sample: CLadderSample,
            family: TokenizerFamily,
            llm: LLMWrapper,
            detached = True,
    ):
        if llm is None:
            raise RuntimeError("`llm` must be provided.")
        if not llm.is_loaded:
            raise RuntimeError("`llm` must be loaded either as context or directly via `llm.load()`")

        edge_index = sample.data.edge_index.to(self.device)
        node_texts = sample.node_names

        with self.profiler.section("S2 // 1-candidate_pairs"):
            candidate_pairs = _generate_candidate_pairs(
                num_nodes=len(node_texts),
                device=self.device
            )

        with self.profiler.section("S2 // 2-PreLLM"):
            H, Z, embeds, attention_mask = self._stage1_p1(
                family=family,
                node_texts=node_texts,
                edge_index=edge_index,
            )
            # intermediate_Z: (num_nodes, llm_dim) in llm_dtype
            intermediate_Z = family.pre_projector(Z)

        if detached:
            return tuple(
                t.detach() for t in
                (candidate_pairs, H, Z, embeds, attention_mask, intermediate_Z)
            )
        return candidate_pairs, H, Z, embeds, attention_mask, intermediate_Z


