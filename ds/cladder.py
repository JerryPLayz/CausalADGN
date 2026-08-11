import re
from dataclasses import dataclass, field
from typing import Optional, Self

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torch_geometric.data import Data
import networkx as nx
import os
from pathlib import Path

_VAR_MAP_RE = re.compile(r"Let\s+(.*?)\.")
_VAR_ENTRY_RE = re.compile(r"(\w+)\s*=\s*([^;]+)")
_EDGE_RE = re.compile(r"(\w+)->(\w+)")

@dataclass
class CLadderSample:
    data: Data
    node_names: list[str]  # semantic discription of nodes
    variable_keys: list[str] # actual variables in the graph
    prompt: str
    label: str  # yes/no
    rung: int  # 0=Associational, 1=Interventional, 2=Counterfactual
    query_type: str
    graph_id: int
    story_id: int
    sample_id: int
    formal_form: str

    @classmethod
    def from_row(cls,
                 data: Data,
                 node_names: list[str],
                 variable_keys: list[str],
                 row
                 ) -> Self:
        return cls(
            data=data,
            node_names=node_names,
            variable_keys=variable_keys,
            prompt=row["prompt"],
            label=row["label"],
            rung=int(row["rung"])-1,  # normalize {1,2,3} -> {0,1,2} for F.cross_entropy
            query_type=row["query_type"],
            graph_id=row["graph_id"],
            story_id=row["story_id"],
            sample_id=int(row["id"]),
            formal_form=row.get("formal_form", ""),
        )


class CLadderDataset(Dataset[CLadderSample]):
    """
    Wraps a list of CLadder Samples for use with torch.utils.data.DataLoader
    Always has batch_size=1.
    """
    def __init__(self, samples: list[CLadderSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> CLadderSample:
        return self.samples[idx]

    @staticmethod
    def as_dataloader(batch: list[CLadderSample]) -> CLadderSample:
        """
        Unpacks the single-element batch list back to a bare CLadderSample.
        batch_size=1 enforced by as_dataloader().
        :param batch: batch to extract from
        :return: CLadderSample
        """
        assert len(batch) == 1, (
            f"CLadderDataset expects batch_size=1, got {len(batch)}"
        )
        return batch[0]

    def as_dataloader(self, shuffle: bool = True) -> DataLoader[CLadderSample]:
        """
        Construct a DataLoader with the correct collate function and batch_size=1 enforced.
        :param shuffle:  Whether to shuffle the dataset each epoch (default True, set False for validation)
        :return: DataLoader[CLadderSample]
        """
        return DataLoader(
            self,
            batch_size=1,
            shuffle=shuffle,
            collate_fn=self.collate
        )

    @classmethod
    def from_samples_split(
            cls,
            samples: list[CLadderSample],
            val_size: float = 0.2,
            seed: int = 42,
            stratify: bool = True
    ) -> tuple[Self, Self]:
        """
        Split a list of CLadderSamples into train and validation datasets.
        Stratification is enabled (to split across rung values).
        :param samples: Full list of CLadderSample objects to split
        :param val_size: Fraction of samples to use for validation
        :param seed: Random seed for reproducibility
        :param stratify: Whether to stratify by rung (default True).
        :return: train_dataset, validation_dataset: Two CLadderDataset objects.
        """
        from sklearn.model_selection import train_test_split
        strat_labels = [s.rung for s in samples] if stratify else None
        train_samples, val_samples = train_test_split(
            samples,
            test_size=val_size,
            random_state=seed,
            stratify=strat_labels
        )
        return cls(train_samples), cls(val_samples)





def _parse_reasoning(
        reasoning: str
) -> tuple[dict[str,str], list[tuple[str, str]]]:
    """
    Parse the reasoning field of the CLadder 1.5 dataset, extracting the variable map (X:semantic meaning, Y: semantic meaning ...), and edges between the variables.
    :param reasoning: The reasoning field of the CLadder 1.5 dataset.
    :return: tuple of (node_map, edges)
    """
    if not isinstance(reasoning, str) or not reasoning.strip():
        return {}, []

    lines = [ln.strip() for ln in reasoning.splitlines() if ln.strip()]
    if all(ln.lower() == "nan" for ln in lines):
        return {}, []

    variable_map: dict[str,str] = {}
    edges: list[tuple[str, str]] = []
    for line in lines:
        # variable mapping line (Let X=...)
        m = _VAR_MAP_RE.search(line)
        if m and not variable_map:
            for var, label in _VAR_ENTRY_RE.findall(m.group(1)):
                variable_map[var.strip()] = label.strip()
            continue

        # Edge Structure Line: only contains symbols and arrows (->)
        if "->" in line and not line.startswith("E[") and not edges:
            edges = _EDGE_RE.findall(line)
            break

    return variable_map, edges


def _build_pyg_graph(
        variable_map: dict[str, str],
        edges: list[tuple[str,str]],
        row: dict
) -> Optional[CLadderSample]:
    """
    Converts parse CLadder SCM data into a PyG data object.
    Node Features (x) are left as a zero placeholder as the pipeline will handle that.
    :param variable_map:
    :param edges:
    :param row:
    :return:
    """
    if not variable_map or not edges:
        return None

    node_keys = list(variable_map.keys())
    node_to_idx = {k: i for i, k in enumerate(node_keys)}

    # Skip edges referencing variables not in the map (e.g. unobserved variables)
    valid_edges = [
        (src, dst)
        for src, dst in edges
        if src in node_to_idx and dst in node_to_idx
    ]

    if not valid_edges:
        return None

    src_idx = [node_to_idx[s] for s, _ in valid_edges]
    dst_idx = [node_to_idx[d] for _, d in valid_edges]

    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    num_nodes = len(node_keys)

    data = Data(
        x=torch.zeros(num_nodes, 1),  # placeholder
        edge_index=edge_index,
        num_nodes=num_nodes
    )

    #print(node_keys, node_to_idx, variable_map)

    # Task Metrics
    smp = CLadderSample.from_row(
        data=data,
        node_names=[variable_map[k] for k in node_keys],
        variable_keys=node_keys,
        row=row
    )

    return smp

# Main loader
RUNG3_QUERY_TYPES = {"ett", "nde", "nie", "det-counterfactual"}


@dataclass
class CLadderLoaderConfig:
    """
    Configuration for the CLadder v1.5 PyG loader.
    rung_filter: If set, only load rows from this rung (1,2,3)
    query_types: If set, further filter to these query type codes. Defaults to all Rung 3 types when rung_filter=3 Pass an empty set to disable and load all query types.
    skip_unparseable: If True (default), silently drop rows whose reasoning fields cannot be parsed into a valid graph. If False, raises on failure.
    """
    rung_filter: Optional[int] = 3
    query_types: Optional[set[str]] = field(default_factory=lambda: RUNG3_QUERY_TYPES)
    skip_unparseable: bool = True

def load_cladder_v1_5(
        config: Optional[CLadderLoaderConfig] = None,
) -> list[CLadderSample]:
    """
    Loads causal-nlp/CLadder (full_v1.5_default split) from HuggingFace and returns a list of PyG Data objects, one per QA sample.
    :param config: A CLadderLoaderConfig object.
    :return:
    """
    if config is None:
        config = CLadderLoaderConfig()

    print("Downloading causal-nlp/CLadder (full_v1.5_default) ...")
    dataset = load_dataset("causal-nlp/CLadder", split="full_v1.5_default", cache_dir=Path(os.getcwd()) / "hf_datasets")

    # Apply rung filter
    if config.rung_filter is not None:
        dataset = dataset.filter(
            lambda row: int(row["rung"]) == config.rung_filter
        )

    if config.query_types:
        dataset = dataset.filter(
            lambda row: row["query_type"] in config.query_types
        )

    print(f"Processing {len(dataset)} rows...")

    graphs: list[Data] = []
    skipped = 0
    for row in dataset:
        variable_map, edges = _parse_reasoning(row["reasoning"])
        data = _build_pyg_graph(variable_map, edges, row)
        if data is None:
            if not config.skip_unparseable:
                raise ValueError(
                    f"Failed to parse {row['id']} due to unparseable reasoning...\n"
                    f"Reasoning:\n{row['reasoning']}"
                )
            skipped += 1
            #print("\n\n" f"Failed to parse {row['id']} due to unparseable reasoning...\n" f"Reasoning:\n{row['reasoning']}")
            continue
        graphs.append(data)
    print(
        f"Loaded {len(graphs)} graphs, skipped {skipped} unparseable rows."
    )
    return graphs
