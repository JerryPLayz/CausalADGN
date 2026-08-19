from __future__ import annotations
from dataclasses import dataclass
from torch.nn import Module
from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import env
from .modules.utils import is_large_model




bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


# Output container (to simplify matters)
@dataclass
class LLMOutputs:
    """
    Structured output from a single LLMWrapper forward pass.
    Attributes:
        - hidden_states: (1, total_len, llm_dim) - full hidden state sequence at layer_index.
        - graph_hidden: (num_nodes, llm_dim) - hidden states at graph node positions only. (Input to PostProjector)
        - last_token_hidden: (llm_dim,) - hidden state of the final sequence token. Input to GateClassifier.
        - next_token_logits: (vocab_size,) - LM head projection of the last_token_hidden. Used for yes/no evaluation (inference=False), to avoid the lm_head projection cost.
    """
    hidden_states: torch.Tensor
    graph_hidden: torch.Tensor
    last_token_hidden: torch.Tensor
    input_embeds: torch.Tensor
    attention_mask: torch.Tensor
    next_token_logits: Optional[torch.Tensor] = None


# noinspection bad-assignment
class LLMWrapper:
    """
    Agnostic wrapper for frozen HuggingFace LLMs in Stage 2
    Not a nn.Module as the LLM is frozen and never trained on. Gradient flows through `input_embeds` only, not LLM parameters.
    """
    _model: Module
    # To append to every prompt before tokenization.
    DEFAULT_YN_SUFFIX = " Answer with yes or no only."

    def __init__(
            self,
            model_id: str,
            device: Union[torch.device, str] = "cuda",
            torch_dtype: torch.dtype = torch.bfloat16,
            layer_index: int = -1
    ):
        """

        :param model_id: HuggingFace model ID
        :param device: Target device
        :param torch_dtype: Model weight dtype, should match the TokenizerFamily's `llm_dtype` (projectors will automatically cast to this dtype and back out again)
        :param layer_index: Hidden state layer to extract (usually -1)
        """
        self.model_id = model_id
        self.device = torch.device(device) if isinstance(device, str) else device
        self.torch_dtype = torch_dtype
        self.layer_index = layer_index
        self._model: Optional[nn.Module] = None
        self._yn_cache: Optional[tuple[set[int], set[int]]] = None

    def load(self) -> "LLMWrapper":
        """
        Load the LLM into memory. Returns self for chaining.
        No-op if already loaded.
        :return: ~`LLMWrapper`
        """
        if self._model is not None:
            return self

        is_large = is_large_model(self.model_id)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            token=env.HF_TOKEN,
            device_map="auto" if is_large else None,
            quantization_config=bnb_config if is_large else None
        ) #.to(self.device)

        if not is_large:
            self._model.to(self.device)

        # Frozen, parameters should not accrue gradients, but still flow through.
        self._model.eval()
        self._model.requires_grad_(False)

        return self

    def unload(self) -> None:
        """Unload the LLM, freeing memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._yn_cache = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __enter__(self) -> "LLMWrapper":
        return self.load()

    def __exit__(self, *_):
        self.unload()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "LLMWrapper: model is not loaded!"
                " Call load() or use a context manager!"
            )

    @staticmethod
    def apply_yn_suffix(
            prompt: str,
            suffix: str = DEFAULT_YN_SUFFIX,
    ) -> str:
        """
        Append yes/no instruction suffix to the end of the prompt string.
        :param prompt: Raw prompt (str) from `CLadderSample.prompt`
        :param suffix: Instruction suffix (default DEFAULT_YN_SUFFIX)
        :return: Modified Prompt String
        """
        return prompt.rstrip() + suffix

    @staticmethod
    def embed_prompt(
            prompt: str,
            tokenizer,
            embed_layer: nn.Embedding,
            device: torch.device,
            suffix: str = DEFAULT_YN_SUFFIX,
    ):
        """
        Apply yes/no suffix, tokenize and embed a prompt string.
        :param prompt: Raw prompt string from `CLadderSample.prompt`
        :param tokenizer: HuggingFace tokenizer for this family.
        :param embed_layer: Frozen nn.Embedding from TokenizerFamily
        :param device: Target device
        :param suffix: Yes/no instruction suffix (default DEFAULT_YN_SUFFIX)
        :return: (prompt_len + suffix_len, llm_dim) FloatTensor
        """
        modified = LLMWrapper.apply_yn_suffix(prompt, suffix)

        tokens = tokenizer(
            modified,
            return_tensors="pt",
            add_special_tokens=True
        ).to(device)

        with torch.no_grad():
            embeds = embed_layer(tokens.input_ids)
        # (1, prompt_len, llm_dim)

        return embeds.squeeze(0)  # (prompt_len + suffix_len, llm_dim)

    @staticmethod
    def assemble_inputs(
            graph_embeds: Optional[torch.Tensor],
            prompt_embeds: torch.Tensor,
            graph_first: bool = True,
    ):
        """
        Assemble input_embeds and attention_mask from graph and prompt embeddings
        No padding as batch_size=1, so attention_mask is all ones.
        :param graph_embeds: Z_proj (num_nodes, llm_dim) from PreProjector. Must be of `llm_dtype`
        :param prompt_embeds: (prompt_len, llm_dim), prompt token embeddings from the frozen embed_layer.
        :param graph_first: If true, graph tokens come first and thus the prompt attends to them under the causal mask. If false, the graph attends to the prompt.
        :return: input_embeds (1, total_len, llm_dim), attention_mask (1, total_len) <- all ones.
        """
        p = prompt_embeds.unsqueeze(0)  # (1, prompt_len, llm_dim)
        if graph_embeds is not None:
            g = graph_embeds.unsqueeze(0)  # (1, num_nodes, llm_dim)

            input_embeds = torch.cat(
                [g, p] if graph_first else [p, g],
                dim=1
            )  # (1, num_nodes + prompt_len, llm_dim)
        else:
            input_embeds = p

        attention_mask = torch.ones(
            1, input_embeds.size(1),
            device=input_embeds.device,
            dtype=torch.long
        )

        return input_embeds, attention_mask

    def forward(
            self,
            input_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            num_nodes: int,
            graph_first: bool = True,
            inference: bool = False,
    ):
        """
        Run the frozen LLM forward pass and extract all Stage 2 relevant outputs in a single call.
        :param input_embeds: (1, total_len, llm_dim), from assemble_inputs()
        :param attention_mask: (1, total_len), from assemble_inputs()
        :param num_nodes: Number of graph nodes, used to slide graph positions from hidden states.
        :param graph_first: Must match `graph_first` used in assemble_inputs()
        :param inference: True during evaluation, False during training.
        :return: ~`LLMOutputs`
        """
        self.require_loaded()

        def _run() -> "LLMOutputs":
            # noinspection PyCallingNonCallable
            outputs = self._model(
                inputs_embeds = input_embeds,
                attention_mask = attention_mask,
                output_hidden_states=True
            )

            # hidden states is a tuple (embedding_output, layer_1, ... layer_N)
            hidden = outputs.hidden_states[self.layer_index]
            # (1, total_len, llm_dim)

            graph_hidden = self._extract_graph_hidden(hidden, num_nodes, graph_first)
            last_token_hidden = self._extract_last_token_hidden(hidden)

            next_token_logits = None
            if inference:
                lm_head = self._model.get_output_embeddings()
                next_token_logits = lm_head(last_token_hidden)
                # (vocab_size,)

            return LLMOutputs(
                hidden_states = hidden,
                graph_hidden = graph_hidden,
                last_token_hidden = last_token_hidden,
                next_token_logits = next_token_logits,
                input_embeds=input_embeds.detach(),
                attention_mask=attention_mask.detach()
            )
        if inference:
            with torch.no_grad():
                return _run()
        else:
            return _run()

    @staticmethod
    def _extract_graph_hidden(
            hidden: torch.Tensor,
            num_nodes: int,
            graph_first: bool
    ) -> torch.Tensor:
        """
        Slice hidden states at graph node positions.

        :param hidden: (1, total_len, llm_dim)
        :param num_nodes: Number of graph nodes
        :param graph_first: Must match `graph_first` used in assemble_inputs()
        :return: (num_nodes, llm_dim)
        """
        if num_nodes == 0:
            return torch.empty(
                0, hidden.size(-1),
                dtype=hidden.dtype,
                device=hidden.device,
            )
        h = hidden.squeeze(0)  # (total_len, llm_dim)
        return h[:num_nodes] if graph_first else h[-num_nodes:]

    @staticmethod
    def _extract_last_token_hidden(
            hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the final token's hidden state.
        :param hidden: (1, total_len, llm_dim)
        :return: (llm_dim)
        """
        return hidden.squeeze(0)[-1]

    @staticmethod
    def build_yn_token_sets(
            tokenizer
    ) -> tuple[set[int], set[int]]:
        """
        Dynamically build yes/no token ID sets from the tokenizer.

        Only single-token encodings are included, as multi-token variants are ambiguous when evaluating the first predicted token.

        :param tokenizer: HuggingFace tokenizer used for this family.
        :return: yes_ids (set[int] of all variants), no_ids (set[int] of all variants)
        """
        yes_variants = ["yes", "Yes", "YES", " yes", " Yes", " YES"]
        no_variants = ["no", "No", "NO", " no", " No", " NO"]

        def _to_ids(variants: list[str]) -> set[int]:
            ids = set()
            for v in variants:
                tokens = tokenizer.encode(v, add_special_tokens=False)
                if len(tokens) == 1:
                    ids.add(tokens[0])
                print(f"[{v}]", end=" ")
            print("")
            return ids

        return _to_ids(yes_variants), _to_ids(no_variants)

    def get_yn_token_sets(self, tokenizer) -> tuple[set[int], set[int]]:
        """
        Get the Yes/No token sets for this model.
        Will cache to ensure no extra computation is required.
        :param tokenizer: HuggingFace tokenizer for this family.
        :return: yes_ids, no_ids; cached after first call
        """
        if self._yn_cache is None:
            self._yn_cache = LLMWrapper.build_yn_token_sets(tokenizer)
        return self._yn_cache

    @staticmethod
    def classify_yn(
            logits: torch.Tensor,
            yes_ids: set[int],
            no_ids: set[int],
    ) -> tuple[str, float, float]:
        """
        Classify via controlled log-probability scoring. See Joshi et al. (2024, arXiv:2406.12158)
        Sums softmax probability mass over all yes/no token variants rather than relying on argmax.
        More robust to tokenization differences across families.
        :param logits: (vocab_size,) from LLMOutputs.next_token_logits
        :param yes_ids: from build_yn_token_sets()
        :param no_ids: from build_yn_token_sets()
        :return: tuple(prediction, confidence, yn_coverage)
            - prediction "yes"/"no"/"abstain" (when the model is definitely not producing a yes/no response => abstain);
            - confidence [0.5-1.0]: probability of predicted class, normalized over yes+no mass only;
            - yn_coverage: total yes+no probability mass, how much of the model's distribution is captured by known tokens.
                - Low coverage (<0.1) suggests the model is not producing a yes/no response at all.
        """
        probs = torch.softmax(logits.float(), dim=-1)

        yes_prob = probs[list(yes_ids)].sum().item() if yes_ids else 0.0
        no_prob = probs[list(no_ids)].sum().item() if no_ids else 0.0

        yn_total = yes_prob + no_prob
        # informative: is the model producing yes/no response rather than some other token.

        if yn_total < 1e-8:
            # model is producing neither, return with 0 confidence
            return "abstain", 0.0, 0.0

        confidence = max(yes_prob, no_prob) / yn_total
        prediction = "yes" if yes_prob > no_prob else "no"
        return prediction, confidence, yn_total

    @torch.no_grad()
    def generate_text(
            self,
            input_embeds: torch.Tensor,
            attention_mask: torch.Tensor,
            tokenizer,
            max_new_tokens: int = 250,
            temperature: float = 0.0,
    ):
        """
        Generate text output for qualitative inspection only.
        Not used for training or evaluation. (But good to check none-the-less for a few cases)
        :param input_embeds: (1, seq_len, llm_dim) from assemble_inputs(
        :param attention_mask:  (1, seq_len)
        :param tokenizer: A TokenizerFamily.tokenizer object
        :param max_new_tokens: Generation length cap (default 250)
        :param temperature: 0.0 = greedy decoding (deterministic, reproducible)
        :return: Generated string, decoded from new tokens only (not the prompt)
        """
        self.require_loaded()
        if input_embeds.dim() == 2:
            input_embeds = input_embeds.unsqueeze(0)
            # -> (1, seq_len, hidden_dim)

        output_ids = self._model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample = temperature > 0.0,
            temperature = temperature if temperature > 0.0 else None,
            pad_token_id = tokenizer.pad_token_id,
        )

        # output_ids includes the input: slice to new tokens only
        new_tokens = output_ids[0, 1:]  # according to PR #21580
        return new_tokens






