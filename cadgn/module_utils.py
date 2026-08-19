import hashlib
from typing import Iterable
from .modules.utils import save_history
from .llm_wrapper import LLMWrapper
from .results.shared import save_sample_results
import pandas as pd
from pathlib import Path
import json
import time

def short_id(text: str, length: int=5):
    """
    Deterministic short ID from a string.
    Uses SHA-256 + base62 encoding
    :param text:
    :param length:
    :return:
    """
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    base = len(CHARS)

    hash_int = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest(), "big")
    n = hash_int % (base ** length)

    result = []
    for _ in range(length):
        result.append(CHARS[n % base])
        n //= base
    return "".join(reversed(result))


def eval_and_save(
        llm: "LLMWrapper",
        tokfam,
        mmd_val,
        history,
        metrics,
        per_sample,
        prompt_samples,
        cls: str="baseline"
):
    t0 = time.time()
    safe_mdl_name = llm.model_id.replace("/", "__")
    save_history(history, path=f"./ablate_results/history/{cls}_{safe_mdl_name}_mmd-{int(mmd_val)}_history.csv")
    print(f"\t>> History saved in {time.time()-t0}s...")
    t0 = time.time()
    #mdf = pd.DataFrame(metrics)
    metrics_path = Path("./ablate_results/metrics")
    metrics_path.mkdir(parents=True, exist_ok=True)
    with open(metrics_path / f"{cls}_{safe_mdl_name}_mmd-{int(mmd_val)}_metrics.json", "w") as f:
        json.dump(metrics, f)
    print(f"\t>> Metrics saved in {time.time()-t0}s...")
    t0 = time.time()
    save_sample_results(
        results=per_sample,
        path=f"./ablate_results/samples/",
        variant=f"{cls}_{safe_mdl_name}_mmd-{int(mmd_val)}",
        epoch=150
    )
    print(f"\t>> Samples saved in {time.time()-t0}s...")
    t0 = time.time()
    # Take a subset of prompts and get text output: see if the qualitative results match statistical expectations...
    print("\t>> Handling Qualitative Gather....")
    o_texts = []
    for i, sample in enumerate(prompt_samples):
        input_embeds = llm.embed_prompt(
            prompt=sample.prompt,
            tokenizer=tokfam.tokenizer,
            embed_layer=tokfam.embed_layer,
            device=llm.device
        )
        _, attention_mask = llm.assemble_inputs(graph_embeds=None, prompt_embeds=input_embeds)
        token_ids = llm.generate_text(
            input_embeds=input_embeds,
            attention_mask=attention_mask,
            tokenizer=tokfam.tokenizer,
        )

        text = tokfam.tokenizer.decode(token_ids, skip_special_tokens=True)
        o_texts.append({
            "id": sample.sample_id,
            "rung": sample.rung,
            "llm_output": text,
            "prompt": sample.prompt,
            "query_type": sample.query_type,
            "reasoning": sample.reasoning,
            "formal_form": sample.formal_form
        })
    output_path = Path("./ablate_results/output")
    output_path.mkdir(parents=True, exist_ok=True)

    o_texts = pd.DataFrame(o_texts)
    o_texts.to_csv(f"./ablate_results/output/{cls}_{safe_mdl_name}_mmd-{int(mmd_val)}_qual.csv")
    print(f"\t>> Output saved in {time.time()-t0}s...")



