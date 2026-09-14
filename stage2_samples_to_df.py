import env
import numpy as np
import pandas as pd

from ds.cladder import load_cladder_v1_5, CLadderLoaderConfig, CLadderSample, CLadderDataset
from cadgn import BaselineSampleResult
import polars as pl
import json
from collections import defaultdict

from pathlib import Path

p = Path("./ablate_results/samples")

do_models = [
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct"
]
model_ablations = ["cadgn", "adgn", "gcn", "gat", "dec"]
mmd_vals = [0, 1]


def get_filepath(abl, modelid, mmd_int, epochs="0040"):
    safe_mdl_name = modelid.replace("/", "__")
    test_path = p / f"{abl}_{safe_mdl_name}_mmd-{mmd_int}_{epochs}_per_sample.json"

    if not test_path.exists():
        print(f"Could not find file with path elements: ({abl}, {modelid} {mmd_int}) for epochs val `{epochs}`")
        return None
    return test_path

filepaths_per_key: dict[tuple, list] = {
    (abl, mdl, int(mmdval)): get_filepath(abl, mdl, int(mmdval))
    for mmdval in [0.0, 1.0]
    for abl in model_ablations
    for mdl in do_models
}

baselines_to_add = {
    ("baseline", mdl, int(mmdval)): get_filepath("baseline", mdl, 0, epochs="0150")
    for mmdval in [0.0, 1.0]
    for mdl in do_models
}

for k,v in baselines_to_add.items():
    filepaths_per_key[k] = v


# Load
samples: defaultdict[tuple, list] = defaultdict(list)
for k, v in filepaths_per_key.items():
    with open(v, "r") as f:
        content = json.load(f)
    for i, r in enumerate(content):
        r["gate_logits"] = []  # not present in saved data, so set None.
        result = BaselineSampleResult(**r)
        samples[k].append(result)

    samples[k].sort(key=lambda r: r.sample_id)
    print(f"({(k[0][:5]):^6} / {k[1]:^35} / {k[2]})", len(samples[k]), [r.sample_id for r in samples[k][:10]])




#
s1_p = Path("./ablate_results/samples/stage1/")
s1_pq = s1_p / "df_graph_pred.parquet"
result = (pl.scan_parquet(s1_pq)
          .select("sample_id")
          .unique()
          .collect()
          .sort("sample_id")
          )

print(result.head(10).to_numpy().flatten())

# Form a parquet file with all the relevant samples together for Stage 2 (s2_BaselineSampleResults.parquet)

rows = []
for k, sample_list in samples.items():
    abl, mdl, mmd_val = k
    for sample in sample_list:
        row = sample.to_dict()
        # inject key components as columns
        row["ablation"]  = abl
        row["model"]     = mdl
        row["mmd_val"]   = mmd_val
        rows.append(row)

df_samples = pd.DataFrame(rows)
print(df_samples.shape)
print(df_samples.head())

df_samples.to_parquet(p / "s2_BaselineSampleResults.parquet", index=False)
print("Saved!")