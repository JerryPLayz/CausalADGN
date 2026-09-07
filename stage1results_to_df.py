import env
import json
import pandas as pd
from pathlib import Path
from models_config import arch, get_stage_configs
from sample_tensors import b64_to_tensor
from ds.cladder import load_cladder_v1_5, CLadderLoaderConfig, CLadderSample
from InProgressInstance import gather_instances, InProgressInstance

import pyarrow as pa
import pyarrow.parquet as pq
"""
This script generates a flattened dataframe and saves it to `stage1_output_graph_path`/df_graph_pred.parquet
This dataframe identifies all copies of the same sample from amongst the ten files (and source sample) and ensures that
all permutations of (mmd_val, ablation, llm) are appropriately flattened such that `pred_embed` and `pred_logits` solely
relates to that combination and not other llms (as was currently stored in the graph_preds.json file).
"""

do_models = [
    "Qwen/Qwen3-4B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.1-8B-Instruct"
]
model_ablations = ["cadgn", "adgn", "gcn", "gat", "dec"]
mmd_vals = [0.0, 1.0]

## Load full dataset for comparison's purposes
dsConfig = CLadderLoaderConfig(rung_filter=None, query_types=None, skip_unparseable=True)

org_ds = load_cladder_v1_5(dsConfig)

## Load models (we need the EncoderHead specifically)
s1c, s2c, evals2c = get_stage_configs(40)
save_path = Path("./saves/ablations/")
save_path.mkdir(parents=True, exist_ok=True)

## Create a container object to store per-sample information
from ds.cladder import CLadderSample, CLadderDataset
from typing import Any, Optional


class SampleExpression:
    def __init__(self,
                 sample_id: int, ablation: str, mmd_val: int,
                 pred_embeds: dict[str, list[float]], pred_logits: dict[str, list[float]], metrics: dict):
        self.sample_id: int = sample_id
        self.ablation: str = ablation
        self.mmd_val: int = mmd_val
        self.pred_embeds: dict[str, list[float]] = pred_embeds
        self.pred_logits: dict[str, list[float]] = pred_logits

        for k, v in metrics.items():
            setattr(self, "m__"+k, v)


class SampleCollection:
    def __init__(self, sample: CLadderSample):
        self.sample = sample
        self.expressions = []
    def add(self, other: SampleExpression):
        if other.sample_id == self.sample.sample_id:
            self.expressions.append(other)

    def __iadd__(self, other: SampleExpression) -> "SampleCollection":
        self.add(other)
        return self

    def to_pandas(self, columns=None, explode_dicts: bool = True) -> pd.DataFrame:
        base_columns = ['sample_id', 'ablation', 'mmd_val', 'pred_embeds', 'pred_logits']
        if columns is None:
            metric_columns = [
                attr for attr in vars(self.expressions[0])
                if attr.startswith('m__')
            ]
            columns = base_columns + metric_columns

        data = []
        for expr in self.expressions:
            row = {col: getattr(expr, col) for col in columns}
            data.append(row)

        df = pd.DataFrame(data)
        if not explode_dicts:
            return df

        try:
            df['_keys'] = df['pred_embeds'].apply(lambda x: list(x.keys()) if isinstance(x, dict) else [])

            df = df.explode('_keys')

            # Rename the exploded column to llm
            df = df.rename(columns={'_keys': 'llm'})

            # Now extract the specific list values for that key
            # This maps: row['pred_embeds']['A'] -> row['pred_embeds']
            df['pred_embeds'] = df.apply(
                lambda row: (row['pred_embeds'].get(row['llm'], []).tolist()
                             if hasattr(row['pred_embeds'].get(row['llm'], []), 'tolist')
                             else row['pred_embeds'].get(row['llm'], []))
                if isinstance(row['pred_embeds'], dict) else row['pred_embeds'],
                axis=1
            )
            df['pred_logits'] = df.apply(
                lambda row: (row['pred_logits'].get(row['llm'], []).tolist()
                             if hasattr(row['pred_logits'].get(row['llm'], []), 'tolist')
                             else row['pred_logits'].get(row['llm'], []))
                if isinstance(row['pred_logits'], dict) else row['pred_logits'],
                axis=1
            )

        except Exception as e:
            print(f"Warning: Could not explode dict columns. Error: {e}")
        return df


train, vald = CLadderDataset.from_samples_split(
    samples=org_ds,
    val_size=0.2,
    stratify=True
)

complete_val = vald.as_dataloader(shuffle=False)

samples: list[SampleCollection] = [SampleCollection(s) for s in complete_val]

stage1_output_graph_path = Path("./ablate_results/samples/stage1/")
stage1_output_graph_path.mkdir(parents=True, exist_ok=True)


for a_id, abl in enumerate(model_ablations):
    for im, mmd_val in enumerate(mmd_vals):
        mmd_int = int(mmd_val)
        mmd_str = str(mmd_int)
        fn = f"{a_id}_{abl}_all_{mmd_str}_s1-graph-preds.json"
        with open(stage1_output_graph_path / fn,"r") as f:
            content = json.load(f)

        # convert the b64 encoded logits/tensors back to numpy
        for i, sample in enumerate(content):
            # (num_nodes, token_count, llm_dim)
            sample["pred_embeds"] = {k: b64_to_tensor(d) for k, d in sample["pred_embeds"].items()}

            # flat adjacency matrix
            sample["pred_logits"] = {k: b64_to_tensor(d) for k, d in sample["pred_logits"].items()}

            k = (sample["sample_id"], mmd_int, abl)
            # Add to a sample
            expr = SampleExpression(
                sample_id=sample["sample_id"],
                ablation=abl,
                mmd_val=mmd_int,
                pred_embeds=sample["pred_embeds"],
                pred_logits=sample["pred_logits"],
                metrics=sample["metrics"]
            )
            for s in samples:
                s += expr
            if i == 0:
                embeds_shape = {k: t.shape for k,t in sample["pred_embeds"].items()}
                edge_logits_shape = {k: t.shape for k,t in sample["pred_logits"].items()}

                print(k, sample["graph_id"], "\n\tEmbeds Shape: ", embeds_shape, "\n\tEdge Logits Shape: ",  edge_logits_shape)


print("Saving in pieces....")
output_path = stage1_output_graph_path / "df_graph_pred.parquet"

writer = None

for si, s in enumerate(samples):
    if not s.expressions:
        print(f"Warning: No expressions for sample {s.sample.sample_id}")
        continue

    if si % 250 == 0:
        print(f"{s.sample.sample_id}[{si}] - Processing...")

    df_chunk = s.to_pandas()
    table = pa.Table.from_pandas(df_chunk, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(
            output_path,
            table.schema,
            compression="zstd"
        )

    writer.write_table(table)
    del df_chunk, table  # free immediately

if writer:
    writer.close()
