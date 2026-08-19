import env
import torch
import transformers as tf
from transformers import BitsAndBytesConfig
#from cadgn import LLMWrapper
from transformers import AutoTokenizer, AutoModelForCausalLM, TokenizersBackend, SentencePieceBackend
from typing import Dict, Tuple, Optional, Union, TypeAlias
from .modules.utils import is_large_model

"""
TO assist with the parameter search libraries and jupyter instances, this cache will ensure we don't have to reload huge 
LLM models between runs, instead a cached version of the tokenizers and embedding layers is stored and used instead.
"""
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
EmbedLayerType: TypeAlias = torch.nn.Embedding
TokenizerType: TypeAlias = Union[tf.TokenizersBackend, tf.SentencePieceBackend]

class ModelCache:
    def __init__(self):
        self._cache: Dict[str, dict] = {}
        # structure: {model_id: {'tokenizer': obj, 'embeddings': nn.Module, 'current_device': torch.device}}

    def get_components(self, model_id: str, torch_dtype: torch.dtype, device: torch.device) -> tuple[TokenizerType, EmbedLayerType]:
        if model_id in self._cache:
            cached: dict = self._cache[model_id]
            cached_device: torch.device = cached['current_device']

            # Is on correct device?
            embed_device: torch.device = next(cached['embeddings'].parameters()).device
            if hash(embed_device) == hash(device):
                return cached['tokenizer'], cached['embeddings']

            # Not on right device (but still in cache)
            print(f"Moving {model_id}'s embed_layer from {cached_device} to {device}")
            cached['embeddings'] = cached['embeddings'].to(device)
            cached['current_device'] = device
            return cached['tokenizer'], cached['embeddings']

        # Do not have in cache, load from source (expensive)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=env.HF_TOKEN)
        # Many LLMs do not ship with a pad token, set one from eos_token.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        is_large = is_large_model(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch_dtype,
            low_cpu_mem_usage=True,
            token=env.HF_TOKEN,
            quantization_config=bnb_config if is_large else None,
            device_map="auto" if is_large else None,
        )

        embeddings = model.get_input_embeddings()
        embeddings.requires_grad_(False)
        embeddings = embeddings.to(device)

        del model

        self._cache[model_id] = {
            'tokenizer': tokenizer,
            'embeddings': embeddings,
            'current_device': device
        }

        return tokenizer, embeddings

    def get_from_llm(self, llmw):
        """
        Gets the tokenizer and embed_layer for a given llm wrapper object (must be load()ed.)
        Does not delete the llmw._model, as it is a context manager.
        :param llmw: LLMWrapper
        :return:
        """
        if llmw.model_id in self._cache:
            cached: dict = self._cache[llmw.model_id]
            cached_device: torch.device = cached['current_device']

            # Is on correct device?
            embed_device: torch.device = next(cached['embeddings'].parameters()).device
            if hash(embed_device) == hash(llmw.device):
                return cached['tokenizer'], cached['embeddings']
            cached['embeddings'] = cached['embeddings'].to(llmw.device)
            cached['current_device'] = llmw.device
            return cached['tokenizer'], cached['embeddings']

        llmw.require_loaded()
        tokenizer: Union[TokenizersBackend, SentencePieceBackend] = AutoTokenizer.from_pretrained(llmw.model_id, token=env.HF_TOKEN)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        embeddings = llmw._model.get_input_embeddings()
        embeddings.requires_grad_(False)
        embeddings = embeddings.to(llmw.device)

        self._cache[llmw.model_id] = {
            'tokenizer': tokenizer,
            'embeddings': embeddings,
            'current_device': llmw.device
        }
        return tokenizer, embeddings





