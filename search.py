from __future__ import annotations
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


from dataclasses import dataclass, asdict
from pathlib import Path
import json
from collections.abc import Callable
from typing import List

import torch
from CADGNTrainer import CADGNTrainer, Stage1Config
from ds.cladder import CLadderDataset
from cadgn import save_history, flush_gpu, CADGNCore, TokenizerFamily




"""
Uses Optuna's TPE sampler for sample-efficient Bayesian optimisation.
MedianPruner kills underperforming trials early to avoid wasting time.

The following parameters remain fixed as a result of limitations in the methodology:
    - max_seq_len
    - mmd_kernel (IMQ is theoretically motivated)
    - reduction  (mean is always correct for this use case)
    - auto_pos_weight (always True given class imbalance in the dataset)
"""

@dataclass
class ArchParams:
    """
    Architectural Parameters samples per trial
    Kept separate from Stage1Config below so factories can consume them cleanly without unpacking a mixed hyperparameter dict.
    """
    # # Shared
    ca_dgn_dim: int = 256

    # # CADGNCore
    # CADGNEncoder
    max_layers: int = 4
    num_iters: int = 3
    epsilon: float = 0.01
    base_gamma: float = 0.1
    encoder_dropout: float = 0.0

    # CADGNDecoder
    decoder_expansion: int = 4
    decoder_dropout: float = 0.1

    # GraphBuilder (all params already covered)

    # # per TokenizerFamily
    # EncoderHead / DecoderHead
    head_encoder_dropout: float = 0.1
    head_decoder_dropout: float = 0.1

    # Projectors
    #projector_expansion: int = 2
    #projector_dropout: float = 0.1

    # Gate Classifier
    #gate_cls_scale: int = 8,
    #gate_cls_dropout: float = 0.1,

    def core_kwargs(self) -> dict:
        """
        Extract kwargs for CADGNCore constructor
        :return: dict of kwargs for CADGNCore
        """
        return {
            "ca_dgn_dim": self.ca_dgn_dim,
            "max_layers": self.max_layers,
            "num_iters": self.num_iters,
            "epsilon": self.epsilon,
            "base_gamma": self.base_gamma,
            "encoder_dropout": self.encoder_dropout,
            "decoder_dropout": self.decoder_dropout,
            "decoder_expansion": self.decoder_expansion,
        }

    def family_kwargs(self) -> dict:
        """
        Extract kwargs for TokenizerFamily.from_pretrained() constructor
        :return: dict of kwargs for `TokenizerFamily.from_pretrained()` constructor
        """
        return {
            "ca_dgn_dim": self.ca_dgn_dim,
            "encoder_dropout": self.head_encoder_dropout,
            "decoder_dropout": self.head_decoder_dropout,
            #"projector_expansion": self.projector_expansion,
            #"projector_dropout": self.projector_dropout,
            #"gate_cls_scale": self.gate_cls_scale,
            #"gate_cls_dropout": self.gate_cls_dropout,

        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArchParams":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SearchConfig:
    """
    Search-level configuration: controls the Optuna study, trial budget, pruning behavior and output locations
    Args:
        n_trials: Number of Optuna trials (default 100)
        search_epochs: Epochs per trial during search (default 20) If kept short, pruning will handle the bad trials early.
        full_epochs: Epochs for final retraining on best config (default 100)
        pruning_warmup: Minimum epochs before a trial can be pruned (Default to 10). Prevent pruning before the model stabilizes
        n_startup_trials: Trials completed before TPE's bayesian model is used (i.e. random exploration) - default 10. Must be < n_trials.
        study_name: Optuna study name
        storage: Optune storage URL. None=In-memory only. Use "sqlite://search.db" to persist across interupted runs, load_if_exists=True means the study will resume automatically.
        results_dir: Root directory for all search outputs.
        device: Target device for all trials.
        seed: Sampler seed for reproducibility.
        max_steps_train: max_steps_per_epoch for training dataloader
        max_steps_val: max_steps_per_epoch for validation dataloader (recommended: ~`max_steps_train` / len(train) * len(vald) )
    """
    n_trials: int = 100
    search_epochs: int = 15
    full_epochs: int = 100
    pruning_warmup: int = 10
    n_startup_trials: int = 10
    study_name: str = "cadgn_stage1_search"
    storage: str = "sqlite://search.db"
    results_dir: str = "./search_results"
    device: str | torch.device = "cuda"
    seed: int = 42
    max_steps_train: int = 550
    max_steps_val: int = 150  # ~ 550/len(train) * len(vald)


class Stage1Search:
    """
    Bayesian hyperparameter and architecture search for Stage 1 training.
    """

    def __init__(
            self,
            core_factory: Callable[[ArchParams], CADGNCore],
            family_factory: Callable[[ArchParams], List[TokenizerFamily]],
            train_dataset: CLadderDataset,
            val_dataset: CLadderDataset,
            search_config: SearchConfig,
    ):
        self.core_factory = core_factory
        self.family_factory = family_factory
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.sc = search_config
        self.results_dir = Path(search_config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # Parameter sampling
    # noinspection bad-argument-type
    def _sample_arch(self, trial: optuna.Trial) -> ArchParams:
        """
        Sample architectural parameters.
        :param trial: optuna.Trial object.
        :return: Architecture parameters (ArchParams) object
        """
        return ArchParams(
            # Shared
            ca_dgn_dim=trial.suggest_categorical(
                "ca_dgn_dim",
                [128, 256, 512, 1024, 2048]
            ),

            # CADGNEncoder
            max_layers=trial.suggest_int("max_layers", 1, 10),
            num_iters=trial.suggest_int("num_iters", 1, 5),
            epsilon=trial.suggest_float("epsilon", 1e-6, 0.5, log=True),
            base_gamma=trial.suggest_float("base_gamma", 1e-6, 0.5, log=True),
            encoder_dropout=trial.suggest_float("encoder_dropout", 0.05, 0.3),

            # CADGNDecoder
            decoder_expansion=trial.suggest_int("decoder_expansion", 2, 8),
            decoder_dropout=trial.suggest_float("decoder_dropout", 0.05, 0.3),

            # Heads (TokenizerFamily)
            head_encoder_dropout=trial.suggest_float("head_encoder_dropout", 0.05, 0.3),
            head_decoder_dropout=trial.suggest_float("head_decoder_dropout", 0.05, 0.3),

            # Projectors (TokenizerFamily)
            #projector_expansion=trial.suggest_int("projector_expansion", 2, 8),
            #projector_dropout=trial.suggest_float("projector_dropout", 0.05, 0.3),

            # GateClassifier
            #gate_cls_scale=trial.suggest_categorical(
            #    "gate_cls_scale",
            #    [2, 4, 8, 16]
            #),
            #gate_cls_dropout=trial.suggest_float("gate_cls_dropout", 0.05, 0.3),
        )

    # noinspection bad-argument-type
    def _sample_training(
            self,
            trial: optuna.Trial,
            arch: ArchParams,
            epochs: int,
    ) -> Stage1Config:
        """
        Sample training parameters.
        :param trial:
        :param arch:
        :param epochs: number of epochs to train for.
        :return:
        """
        return Stage1Config(
            # Optimizer
            lr=trial.suggest_float("lr", 1e-6, 1e-2, log=True),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),

            # Loss Weights
            #w_nodes=trial.suggest_float("w_nodes", 0.1, 2.0),
            #w_edges=trial.suggest_float("w_edges", 0.1, 2.0),
            #w_norm=trial.suggest_float("w_norm", 0.001, 1.0, log=True),
            #w_mmd=trial.suggest_float("w_mmd", 0.01, 2.0),
            #w_mse=trial.suggest_float("w_mse", 0.01, 2.0),
            #w_cosine=trial.suggest_float("w_cosine", 0.01, 2.0),

            # Norm Loss
            #norm_w_preserve=trial.suggest_float("norm_w_preserve", 0.1, 2.0),
            #norm_w_floor=trial.suggest_float("norm_w_floor", 0.1, 2.0),

            # Training dynamics
            grad_accum_steps = trial.suggest_int("grad_accum_steps", 1, 16),

            # Fixed
            epochs=epochs,
            max_steps_per_epoch=self.sc.max_steps_train,
            max_steps_per_epoch_val=self.sc.max_steps_val,
        )

    # Trial objective
    def _objective(self, trial: optuna.trial.Trial) -> float:
        """
        Single trial objective.
        Samples architecture and training config for search_epochs, and reports val/loss after each epoch for pruning.
        :param trial:
        :return:
        """
        arch: ArchParams = self._sample_arch(trial)
        config: Stage1Config = self._sample_training(trial, arch, self.sc.search_epochs)

        core = self.core_factory(arch)
        families = self.family_factory(arch)

        trainer = CADGNTrainer(
            core=core,
            families=families,
            device=self.sc.device,
            profile=False
        )

        # Construct our dataloaders from the datasets
        train_d = self.train_dataset.as_dataloader(
            max_steps_per_epoch=self.sc.max_steps_train,
            seed=self.sc.seed,
        )

        val_d = self.val_dataset.as_dataloader(
            shuffle=False,
            max_steps_per_epoch=self.sc.max_steps_val,
            seed=self.sc.seed,
        )

        def pruning_callback(epoch: int, val_loss: float) -> None:
            trial.report(val_loss, epoch)
            if (
                epoch >= self.sc.pruning_warmup
                and trial.should_prune()
            ):
                raise optuna.TrialPruned()

        try:
            history = trainer.train_stage1(
                config=config,
                train_dataloader=train_d,
                val_dataloader=val_d,
                pruning_callback=pruning_callback
            )
        except optuna.TrialPruned:
            raise
        finally:
            # Clean up
            del trainer, core, families
            flush_gpu()

        val_losses = history.get("val/loss", [])
        # noinspection bad-assignment
        best_val_loss: float = min(val_losses) if val_losses else float("inf")

        # Save per-trial history for post-hoc analysis
        trial_dir = self.results_dir / "trials" / f"trial_{trial.number:05d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        save_history(history=history, path=trial_dir / "history.csv")

        # Save arch params alongside history for reconstruction later
        with open(trial_dir / "arch_params.json", "w") as f:
            json.dump(arch.to_dict(), f, indent=4)

        return best_val_loss

    # Study orchestration
    def run(self) -> optuna.Study:
        """
        Run the fulls earch.
        Returns the completed Optuna study:
            - study.best_params: best combined arch + training params
            - study.best_value: best val/loss achieved
            - study.trials: all trial objects
        :return:
        """
        study = optuna.create_study(
            study_name=self.sc.study_name,
            direction="minimize",
            sampler=TPESampler(
                seed=self.sc.seed,
                n_startup_trials=self.sc.n_startup_trials,
            ),
            pruner=MedianPruner(
                n_startup_trials=self.sc.n_startup_trials,
                n_warmup_steps=self.sc.pruning_warmup
            ),
            storage=self.sc.storage,
            load_if_exists=True,
        )

        study.optimize(
            self._objective,
            n_trials=self.sc.n_trials,
            n_jobs=1
        )

        self._save_search_results(study)
        self._train_best(study)

        return study

    # Post-search
    def _save_search_results(self, study: optuna.Study) -> None:
        """
        Save full trials dataframe, best params, and best arch params to results_dir
        :param study: the study object to save
        """
        # All trials
        df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
        df.to_csv(self.results_dir / "all_trials.csv", index=False)

        # Best combined parameters
        with open(self.results_dir / "best_params.json", "w") as f:
            json.dump(study.best_params, f, indent=4)

        # Best arch params separately - for clean factory reconstruction
        arch = ArchParams.from_dict(study.best_params)
        with open(self.results_dir / "best_arch_params.json", "w") as f:
            json.dump(arch.to_dict(), f, indent=4)

        print("\n\n\t\t\t Search Complete!")
        print(f"\tBest val/loss : {study.best_value:.6f}")
        print(f"\tBest trial    : {study.best_trial.number}")
        print(f"\tBest arch     : {arch}")
        print(f"\tResults saved : {self.results_dir}")

    def _train_best(self, study: optuna.Study) -> None:
        """
        Reconstruct the best architcture and training config, retrain from scratch for full_epochs, and save the final checkpoint.
        :param study:
        :return:
        """
        print(f"Retraining best config for {self.sc.full_epochs} epochs...")

        arch = ArchParams.from_dict(study.best_params)

        # Reconstruct Stage1Config from best_params

        training_fields = Stage1Config.__dataclass_fields__.keys()
        training_kwargs = {
            k: v for k,v in study.best_params.items()
            if k in training_fields
        }

        config = Stage1Config(
            **training_kwargs,
            epochs=self.sc.full_epochs,
            max_steps_per_epoch=self.sc.max_steps_train,
            max_steps_per_epoch_val=self.sc.max_steps_val
        )

        core: CADGNCore = self.core_factory(arch)
        families: list[TokenizerFamily] = self.family_factory(arch)

        trainer = CADGNTrainer(
            core=core,
            families=families,
            device=self.sc.device,
            profile=True
        )

        best_dir = self.results_dir / "best_model"
        best_dir.mkdir(parents=True, exist_ok=True)

        train_d = self.train_dataset.as_dataloader(
            max_steps_per_epoch=self.sc.max_steps_train,
            seed=self.sc.seed,
        )

        val_d = self.val_dataset.as_dataloader(
            max_steps_per_epoch=self.sc.max_steps_val,
            seed=self.sc.seed,
            shuffle=False
        )

        history = trainer.train_stage1(
            config=config,
            train_dataloader=train_d,
            val_dataloader=val_d,
            checkpoint_dir=best_dir / "checkpoints",
            checkpoint_every=10
        )

        save_history(history=history, path=best_dir / "history.csv")

        # Save the final model
        core.save(path=best_dir / "core", model_id="best")
        for fam in families:
            safe_name = fam.model_id.replace("/", "-")
            fam.save(path=best_dir / "families", model_id=f"{safe_name}-best")

        print(f"\tBest Model saved to {best_dir}")

