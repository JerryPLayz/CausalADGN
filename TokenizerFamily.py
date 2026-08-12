from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Iterable, Literal, Any
import re

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TokenizersBackend, SentencePieceBackend

from cadgn import EncoderHead, DecoderHead, Projector, GateClassifier, short_id

from stager import Staged

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class TokenizerFamily(nn.Module, Staged):
    """
    Container for all components specific to one LLM tokenizer family.
    This is not a pipeline object, simply a structure to make checkpointing simple.
    """
    SAVE_LOAD_PREFIX = "TokFam"

    def __init__(
            self,
            model_id: str,
            tokenizer: TokenizersBackend | SentencePieceBackend,
            embed_layer: nn.Embedding,
            encoder_head: EncoderHead,
            decoder_head: DecoderHead,
            pre_projector: Projector,
            post_projector: Projector,
            gate_classifier: GateClassifier,
            llm_dim: int,
            ca_dgn_dim: int,
            max_seq_len: int,
            encoder_dropout: float = 0.1,
            decoder_dropout: float = 0.1,
            projector_expansion: int = 2,
            projector_dropout: float = 0.1,
            gate_cls_scale: int = 8,
            gate_cls_dropout: float = 0.1,
            llm_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()

        # Plain attribute
        self.model_id = model_id
        self.tokenizer = tokenizer

        # Registered submodules (already included in state_dict())
        self.embed_layer = embed_layer
        self.encoder_head = encoder_head
        self.decoder_head = decoder_head
        self.pre_projector = pre_projector
        self.post_projector = post_projector
        self.gate_classifier = gate_classifier

        self.max_seq_len = max_seq_len
        # Short model_id : sh_id
        self.sh_id = short_id(self.model_id, length=5)

        # Stored verbatim
        self._config = {
            "model_id": model_id,
            "llm_dim": llm_dim,
            "ca_dgn_dim": ca_dgn_dim,
            "max_seq_len": max_seq_len,
            "encoder_dropout": encoder_dropout,
            "decoder_dropout": decoder_dropout,
            "projector_expansion": projector_expansion,
            "projector_dropout": projector_dropout,
            # Needed to reconstruct nn.Embedding on load() without the LLM
            "vocab_size": embed_layer.num_embeddings,
            "padding_idx": embed_layer.padding_idx,
            "gate_cls_scale": gate_cls_scale,
            "gate_cls_dropout": gate_cls_dropout,
            # LLM dtype
            "llm_dtype": str(llm_dtype).replace("torch.", ""),
            "pre_projector_dtype": str(llm_dtype).replace("torch.", ""),
            "post_projector_dtype": "float32",

        }

    @classmethod
    def from_pretrained(
            cls,
            model_id: str,
            ca_dgn_dim: int,
            max_seq_len: int,
            encoder_dropout: float = 0.1,
            decoder_dropout: float = 0.1,
            projector_expansion: int = 2,
            projector_dropout: float = 0.1,
            gate_cls_scale: int = 8,
            gate_cls_dropout: float = 0.1,
            device: str | torch.device = "cuda",
            torch_dtype: torch.dtype = torch.bfloat16,
    ) -> TokenizerFamily:
        """
        Constructs a TokenizerFamily object from a HuggingFace model.

        loads the full LLM transiently on CPU to extract the embedding table, then immediately discards it. Only the tokenizer and embedding layer are retained in memory.
        All subsequent use of this TokenizerFamily requires no LLM (until Stage 2 training & inference)
        :param model_id: HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B")
        :param ca_dgn_dim:  CA-DGN hidden dimension size
        :param max_seq_len:  Maximum tokenized sequence length per node
        :param encoder_dropout:  Dropout in EncoderHead (default 0.1)
        :param decoder_dropout:  Dropout in DecoderHead (default 0.1)
        :param projector_expansion:  Hidden dim scalar multiple for projectors.
        :param projector_dropout:  Dropout in pre/post Projectors (default 0.1)
        :param gate_cls_scale:  Scalar divider for gate layers, to compress down by in dynamic n-layers (default 8)
        :param gate_cls_dropout:  Dropout between layers in the gate classifier (default 0.1)
        :param device:  Target Device for all nn.Modules
        :param torch_dtype: LLM weight dtype. bfloat16 recommended, as it matches typical LLM release formats and halves memory vs float32 for the embedding table.
        :return: `TokenizerFamily` object
        """
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Many LLMs do not ship with a pad token, which we require for batching
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Extract Embedding Layer from LLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            low_cpu_mem_usage=True
        )

        embed_layer = model.get_input_embeddings()
        embed_layer.requires_grad_(False)
        embed_layer = embed_layer.to(device)

        llm_dim = embed_layer.weight.shape[1]
        llm_dtype = embed_layer.weight.dtype

        # Discord the rest of the model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Build the encoder and decoder heads
        encoder_head = EncoderHead(
            llm_dim=llm_dim,
            ca_dgn_dim=ca_dgn_dim,
            max_seq_len=max_seq_len,
            dropout=encoder_dropout,
        ).to(device)

        decoder_head = DecoderHead(
            llm_dim=llm_dim,
            ca_dgn_dim=ca_dgn_dim,
            max_seq_len=max_seq_len,
            dropout=decoder_dropout,
        ).to(device)

        pre_projector = Projector(
            d_from=ca_dgn_dim,
            d_to=llm_dim,
            expansion=projector_expansion,
            dropout=projector_dropout,
            output_dtype=llm_dtype
        ).to(device)

        post_projector = Projector(
            d_from=llm_dim,
            d_to=ca_dgn_dim,
            expansion=projector_expansion,
            dropout=projector_dropout,
            output_dtype=torch.float32
        ).to(device)

        gate_classifier = GateClassifier(
            llm_dim=llm_dim,
            scale=gate_cls_scale,
            dropout=gate_cls_dropout,
        ).to(device)

        return cls(
            model_id=model_id,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            encoder_head=encoder_head,
            decoder_head=decoder_head,
            pre_projector=pre_projector,
            post_projector=post_projector,
            gate_classifier=gate_classifier,
            llm_dim=llm_dim,
            ca_dgn_dim=ca_dgn_dim,
            max_seq_len=max_seq_len,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
            projector_expansion=projector_expansion,
            projector_dropout=projector_dropout,
            gate_cls_scale=gate_cls_scale,
            gate_cls_dropout=gate_cls_dropout,
            llm_dtype=llm_dtype,
        )

    # Persistence on Disk (save/load)

    def _save_extra(self, path: Path, model_id: str, *args, **kwargs):
        self.tokenizer.save_pretrained(path / f"{self.SAVE_LOAD_PREFIX}_{model_id}_tokenizer")

    @classmethod
    def load(
            cls,
            path: str|Path,
            model_id: str,
            map_location: Optional[str | torch.device] = None,
    ) -> TokenizerFamily:
        """
        Reconstruct a TokenizerFamily from a directory produced by save()
        :param path: directory containing the config.json, tokenizer/ and weights.pt files & directory for a particular `model_id` using the prefix `prefix`.
        :param model_id: To help disambiguate models put in the same folder, the model id is used to name the files/folder
        :param prefix: a common prefix to distinguish these kinds of modules from others in the suite.
        :param map_location: Passed to torch.load. Use to laod onto a different device than the checkpoint was saved from. (e.g. `cpu`, `cuda:1`)
        :return: TokenizerFamily with all weights restored.
        """

        path = Path(path)

        with open(path / f"{cls.SAVE_LOAD_PREFIX}_{model_id}_config.json", "r") as f:
            config = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(path / f"{cls.SAVE_LOAD_PREFIX}_{model_id}_tokenizer")

        # Reconstruct modules from config dimensions
        embed_layer = nn.Embedding(
            config["vocab_size"],
            config["llm_dim"],
            padding_idx=config["padding_idx"],
        )
        embed_layer.requires_grad_(False)

        encoder_head = EncoderHead(
            llm_dim=config["llm_dim"],
            ca_dgn_dim=config["ca_dgn_dim"],
            max_seq_len=config["max_seq_len"],
            dropout=config["encoder_dropout"],
        )

        decoder_head = DecoderHead(
            llm_dim=config["llm_dim"],
            ca_dgn_dim=config["ca_dgn_dim"],
            max_seq_len=config["max_seq_len"],
            dropout=config["decoder_dropout"],
        )

        pre_projector = Projector(
            d_from=config["ca_dgn_dim"],
            d_to=config["llm_dim"],
            expansion=config["projector_expansion"],
            dropout=config["projector_dropout"],
            output_dtype=_DTYPE_MAP.get(config["pre_projector_dtype"], torch.bfloat16)
        )

        post_projector = Projector(
            d_from=config["llm_dim"],
            d_to=config["ca_dgn_dim"],
            expansion=config["projector_expansion"],
            dropout=config["projector_dropout"],
            output_dtype=_DTYPE_MAP.get(config["post_projector_dtype"], torch.bfloat16)
        )

        gate_classifier = GateClassifier(
            llm_dim=config["llm_dim"],
            scale=config["gate_cls_scale"],
            dropout=config["gate_cls_dropout"],
        )

        # Assemble and Restore the full object
        instance = cls(
            model_id=model_id,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            encoder_head=encoder_head,
            decoder_head=decoder_head,
            pre_projector=pre_projector,
            post_projector=post_projector,
            gate_classifier=gate_classifier,
            llm_dim=config["llm_dim"],
            ca_dgn_dim=config["ca_dgn_dim"],
            max_seq_len=config["max_seq_len"],
            encoder_dropout=config["encoder_dropout"],
            decoder_dropout=config["decoder_dropout"],
            projector_expansion=config["projector_expansion"],
            projector_dropout=config["projector_dropout"],
            gate_cls_scale=config["gate_cls_scale"],
            gate_cls_dropout=config["gate_cls_dropout"],
        )

        state = torch.load(
            path / f"{cls.SAVE_LOAD_PREFIX}_{model_id}_weights.pt",
            map_location=map_location,
            weights_only=True
        )

        instance.load_state_dict(state)
        return instance

    def configure_stage1(self):
        """
        Configure gradient flow for Stage 1 Training
        Trainable: encoder_head, decoder_head
        Frozen: embed_layer (always), pre_projector, post_projector, gate_classifier
        :return: None
        """
        # Always
        self.embed_layer.requires_grad_(False)

        # Trainable in Stage 1
        self.encoder_head.requires_grad_(True)
        self.decoder_head.requires_grad_(True)

        # Freeze Stage 2 Components if already registered
        self.pre_projector.requires_grad_(False)
        self.post_projector.requires_grad_(False)
        self.gate_classifier.requires_grad_(False)

    def configure_stage2(self):
        """
        Configure gradient flow for Stage 2 Training.
        Trainable: pre_projector, post_projector, gate_classifier
        Frozen: embed_layer (always), encoder_head, decoder_head
        :return: None
        """

        # Always
        self.embed_layer.requires_grad_(False)

        # Freeze Stage 1 Components
        self.encoder_head.requires_grad_(False)
        self.decoder_head.requires_grad_(False)

        # Trainable in Stage 2
        self.pre_projector.requires_grad_(True)
        self.post_projector.requires_grad_(True)
        self.gate_classifier.requires_grad_(True)

    @property
    def __config__(self) -> dict[str, Any]:
        return self._config











