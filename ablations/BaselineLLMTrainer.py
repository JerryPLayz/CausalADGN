import torch
from CADGNTrainer import Stage2Config
from cadgn import GateClassifier, Profiler, TokenizerFamily, LLMWrapper, LLMOutputs, DTYPE_MAP
from contextlib import nullcontext
from ds.cladder import CLadderSample
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassF1Score, BinaryF1Score

from typing import Union, Optional
from cadgn import BaselineSampleResult


class BaselineLLMTrainer:
    """
    Ablation baseline: LLM only, no graph injection
    Runs: prompt -> LLM -> last token hidden -> GateClassifier
                        -> next token logits -> yes/no scoring (constrained log probability)
    """
    def __init__(
            self,
            gate_classifier: GateClassifier,
            device: Union[str, torch.device] = "cuda"
    ):
        self.gate_classifier = gate_classifier.to(
            torch.device(device) if isinstance(device, str) else device
        )
        self.device = device
        self.profiler = Profiler(enabled=True)

    def _build_optimizer(self, config: Stage2Config) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            [p for p in self.gate_classifier.parameters() if p.requires_grad],
            lr = config.lr,
            weight_decay = config.weight_decay,
        )

    def train(
            self,

            config: Stage2Config,
            family: TokenizerFamily,  # for tokenizer + embed_layer only
            train_dataloader: DataLoader[CLadderSample],
            val_dataloader: Optional[DataLoader[CLadderSample]],
            llm: LLMWrapper
    ) -> dict[str, list[float]]:
        """
        Train GateClassifier only, no graph, no projectors.
        :param config: Stage2Config (used ONLY to get `lr`, `weight_decay`, `epochs`, `grad_accum_steps`, `grad_clip`)
        :param family: TokenizerFamily (tokenizer + embed_layer only)
        :param train_dataloader: DataLoader for training data
        :param val_dataloader: Optional validation dataloader
        :param llmw: Optional pre-loaded wrapper. If None, creates and manages a LLMWrapper context internally.
        :return: metrics, per_sample_per_epoch
        """
        # self.gate_classifier.train()
        self.gate_classifier.requires_grad_(True)
        optimizer = self._build_optimizer(config)
        history: dict[str, list[float]] = {}
        per_sample_per_epoch = []

        llm_dtype = DTYPE_MAP.get(family.__config__["llm_dtype"], torch.bfloat16)

        print(f"Baseline training (no graph) for {family.model_id}:")

        #with ctx as llm:
        yes_ids, no_ids = llm.get_yn_token_sets(family.tokenizer)

        for epoch in range(config.epochs):
            #print(f"Epoch {epoch:04d} [", end="")
            self.gate_classifier.train()
            epoch_metrics: dict[str, list[float]] = {}
            optimizer.zero_grad()

            epoch_metrics.setdefault("L_gate", [])
            epoch_metrics.setdefault("loss", [])

            with self.profiler.section("BL // Epoch"):
                for step_idx, sample in enumerate(train_dataloader):
                    with self.profiler.section("BL // Step"):
                        _, gate_logits, _ = self._step(
                            sample=sample,
                            family=family,
                            llm=llm,
                            inference=False,
                        )
                    # GateClassifier loss; training signal
                    with self.profiler.section("BL // GateCls_CalcLoss"):
                        L_gate = GateClassifier.calc_loss(
                            gate_cls_logits=gate_logits,
                            rung_t=sample.rung,
                            device=self.device
                        )

                    loss = (L_gate * config.w_gate) / config.grad_accum_steps
                    loss.backward()

                    is_accum_step = (step_idx + 1) % config.grad_accum_steps == 0
                    is_last_step = (
                        config.max_steps_per_epoch is not None
                        and step_idx + 1 >= config.max_steps_per_epoch
                    )
                    if is_accum_step or is_last_step:
                        if config.grad_clip > 0.0:
                            nn.utils.clip_grad_norm_(
                                list(self.gate_classifier.parameters()),
                                config.grad_clip,
                            )
                        optimizer.step()
                        optimizer.zero_grad()
                    epoch_metrics["L_gate"].append(L_gate.item())
                    epoch_metrics["loss"].append(loss.item())
                #print(".]")
            # Post Epoch...
            epoch_avg = {
                k : sum(v) / len(v) for k, v in epoch_metrics.items()
            }

            if val_dataloader is not None:
                with self.profiler.section("BL // Val"):
                    val_avg, val_per_sample = self._eval(
                        val_dataloader=val_dataloader,
                        config=config,
                        family=family,
                        llm=llm,
                        yes_ids=yes_ids,
                        no_ids=no_ids,
                    )
                # Restore
                self.gate_classifier.train()
                self.gate_classifier.requires_grad_(True)

                for k,v in val_avg.items():
                    epoch_avg[f"val/{k}"] = v

                # Store per-sample values.
                per_sample_per_epoch.append(val_per_sample)
                #history.setdefault("_per_sample", []).append(val_per_sample)

            for k, v in epoch_avg.items():
                history.setdefault(k, []).append(v)

            if self.profiler.enabled:
                print(
                    f"Epoch {epoch:>4d} | "
                    f"L_gate: {epoch_avg.get('L_gate', float('nan')):.6f} | "
                    f"Val Acc: {epoch_avg.get('val/accuracy', float('nan')):.4f} | "
                    f"{self.profiler.get_latest_n('BL // Epoch'):.2f}s (val: +{self.profiler.get_latest_n('BL // Val'):.2f}s)"
                )
            else:
                print(f"{epoch}", end=", ")
        if self.profiler.enabled:
            print("Baseline LLM Profiling Summary:")
            print(self.profiler.summary(sort_by="total"))
        return history, per_sample_per_epoch

    def _step(
            self,
            sample: CLadderSample,
            family: TokenizerFamily,
            llm: LLMWrapper,
            inference: bool = False
    ) -> tuple[dict, torch.Tensor, LLMOutputs]:
        """
        Prompt-only forward pass, no graph injection.
        :param sample:  CLadderSample from a DataLoader
        :param family:  TokenizerFamily; tokenizer and embed_layer only (no need to train Stage 1)
        :param llm:  Loaded LLMWrapper
        :param inference: If True, computes next_token_logits for yes/no scoring (do this during evaluation)
        :return: (metrics {} - L_gate computed by caller, gate_logits: (3, ) raw logits, not detached, llm_outputs: LLMOutputs; next_token_logits is populated if inference=True)
        """
        llm.require_loaded()
        with self.profiler.section("BL // 01-EmbedPrompt"):
            prompt_embeds = LLMWrapper.embed_prompt(
                prompt = sample.prompt,
                tokenizer = family.tokenizer,
                embed_layer=family.embed_layer,
                device=self.device,
            )  # (prompt_len, llm_dim)

        # No graph, prompt only, unsqueeze for batch dim
        input_embeds = prompt_embeds.unsqueeze(0)
        # (1, prompt_len, llm_dim)
        attention_mask = torch.ones(
            1, input_embeds.size(1),
            device=self.device,
            dtype=torch.long,
        )

        with self.profiler.section("BL // 02-LLMForward"):
            llm_outputs = llm.forward(
                input_embeds = input_embeds,
                attention_mask = attention_mask,
                num_nodes=0,
                graph_first=False,
                inference=inference,
            )

        with self.profiler.section("BL // 03-GateClsInfer"):
            gate_logits = self.gate_classifier(
                llm_outputs.last_token_hidden.unsqueeze(0),
            ).squeeze(0)

        return {}, gate_logits, llm_outputs

    @torch.no_grad()
    def _eval(
            self,
            val_dataloader: DataLoader[CLadderSample],
            config: Stage2Config,
            family: TokenizerFamily,
            llm: LLMWrapper,
            yes_ids: set[int],
            no_ids: set[int],
    ) -> tuple[dict[str, float], list[BaselineSampleResult]]:
        """
        An evaluation loop
        :param val_dataloader:
        :param config:
        :param family:
        :param llm: the LLMWrapper object on which to operate.
        :param yes_ids:
        :param no_ids:
        :return: (metrics [aggregated], per_sample list)
        """
        self.gate_classifier.eval()
        self.gate_classifier.requires_grad_(False)

        all_metrics: dict[str, list[float]] = {}
        per_sample: list[BaselineSampleResult] = []

        gate_f1 = MulticlassF1Score(num_classes=3, average="none").to(self.device)
        gate_f1_macro = MulticlassF1Score(num_classes=3, average="macro").to(self.device)
        yn_f1 = BinaryF1Score().to(self.device)
        abstain_count = 0
        total_count = 0

        for step_idx, sample in enumerate(val_dataloader):
            total_count += 1

            _, gate_logits, llm_outputs = self._step(
                sample = sample,
                family = family,
                llm = llm,
                inference = True
            )

            # GateClassifier
            L_gate = GateClassifier.calc_loss(
                gate_cls_logits = gate_logits,
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
                if prediction != "abstain"
                else 0.5
            )

            # Construct the per-sample record
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

            # Metrics
            all_metrics.setdefault(f"accuracy_rung{sample.rung}", []).append(yn_correct)
            all_metrics.setdefault(f"yn_coverage_rung{sample.rung}", []).append(yn_coverage)
            all_metrics.setdefault(f"abstain_rung{sample.rung}", []).append(float(prediction == "abstain"))
            all_metrics.setdefault("L_gate", []).append(L_gate.item())
            all_metrics.setdefault("accuracy", []).append(yn_correct)
            all_metrics.setdefault("confidence", []).append(confidence)
            all_metrics.setdefault("yn_coverage", []).append(yn_coverage)
            all_metrics.setdefault("gate_correct", []).append(gate_correct)

        # Aggregate
        result = {k: sum(v) / len(v) for k, v in all_metrics.items()}

        gate_f1_per_class = gate_f1.compute()
        rung_names = ["assoc", "interv", "counterfact"]
        for i, name in enumerate(rung_names):
            result[f"gate_f1_{name}"] = gate_f1_per_class[i].item()

        result["gate_f1_macro"] = gate_f1_macro.compute().item()
        result["yn_f1"] = (
            yn_f1.compute().item() if total_count > abstain_count else 0.0
        )
        result["abstain_rate"] = abstain_count / total_count if total_count > 0 else 0.0

        for rung in range(3):
            for metric in ("accuracy", "yn_coverage", "abstain"):
                key = f"{metric}_rung{rung}"
                if key not in result:
                    result[key] = float("nan")
        return result, per_sample





