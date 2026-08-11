from __future__ import annotations
import math
from typing import Optional, Iterable, TypedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_networkx
import networkx as nx

# Diameter Utils to help us compute a graph's diameter.


def single_graph_diameter(
        edge_index: torch.Tensor,
        num_nodes: int
) -> int:
    """
    BFS-based diameter for a single graph.
    Uses undirected diameter as an upper bound, conservative and safe
    A directed graph may need fewer layers than an undirected one in practice, but over-estimating is preferable with respect to receptive field.
    :param edge_index:
    :param num_nodes:
    :return: int - diameter of graph
    """
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(data, to_undirected=True)

    if G.number_of_nodes() == 0:
        return 0

    if not nx.is_connected(G):
        largest = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest)

    return nx.diameter(G)


def batch_diameter(
        edge_index: torch.Tensor,
        num_nodes: int,
        batch_vector: Optional[torch.Tensor] = None,
        precomputed: Optional[torch.Tensor] = None,
) -> int:
    """
    Computes the maximum diameter of all graphs in batch.

    This implementation permits precomputed diameters if available (fast O(1) lookup). Otherwise, it will fallback to live BFS computation O(n(n+m)) per graph.
    :param edge_index: (2, num_edges) global edge index (batched)
    :param num_nodes:  total nodes across all graphs in batch.
    :param batch_vector: (num_nodes,) maps node to graph index. None implies a single graph
    :param precomputed: (num_graphs,) LongTensor of pre-cached diameters. Stored as data.diameter in the Data object.
    :return: Maximum diameter across the batch - used as num_layers in CADGN
    """
    if precomputed is not None:
        return int(precomputed.max().item())

    if batch_vector is None:
        return single_graph_diameter(edge_index, num_nodes)

    num_graphs = int(batch_vector.max().item()) + 1
    diameters = []

    for g_idx in range(num_graphs):
        # Mask for nodes belonging to graph g_idx
        node_mask = batch_vector == g_idx
        node_ids = node_mask.nonzero(as_tuple=True)[0]

        # Remap global node indices to local [0, n_g)
        local_map = torch.full(
            (num_nodes,), -1, dtype=torch.long, device=edge_index.device
        )
        local_map[node_ids] = torch.arange(
            len(node_ids), device=edge_index.device
        )

        # Mask edges belonging to this graph
        src, dst = edge_index
        edge_mask = node_mask[src] & node_mask[dst]
        local_edge = local_map[edge_index[:, edge_mask]]
        diameters.append(single_graph_diameter(local_edge, len(node_ids)))

    return max(diameters)


def precompute_diameter(dataset):
    """
    Pre-cache diameter for every graph in a dataset.
    Call once before training, ensuring that the expensive task is handled first.
    """

    for data in dataset:
        if not hasattr(data, 'diameter') or data.diameter is None:
            d = single_graph_diameter(data.edge_index, data.num_nodes)
            data.diameter = torch.tensor(d, dtype=torch.long)

