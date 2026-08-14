from __future__ import annotations
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ds.cladder import CLadderSample

_EDGE_STYLE: dict[str, dict] = {
    "tp": {"color": "#2ecc71", "style": "solid",  "label": "True Positive (correct)"},
    "fp": {"color": "#e74c3c", "style": "solid",  "label": "False Positive (spurious)"},
    "fn": {"color": "#f39c12", "style": "dashed", "label": "False Negative (missing)"},
}

def _classify_edges(
    true_edge_index: torch.Tensor,  # (2, num_true)
    pred_edge_index: torch.Tensor,  # (2, num_pred)
) -> dict[str, list[tuple[int, int]]]:
    """
    Partitions edges into TP, FP, FN sets.
    """
    def to_set(ei: torch.Tensor) -> set[tuple[int, int]]:
        if ei.numel() == 0:
            return set()
        return {(ei[0, k].item(), ei[1, k].item()) for k in range(ei.size(1))}

    true_set = to_set(true_edge_index)
    pred_set = to_set(pred_edge_index)

    return {
        "tp": sorted(true_set & pred_set),  # correctly predicted
        "fp": sorted(pred_set - true_set),  # spurious
        "fn": sorted(true_set - pred_set),  # missed
    }


def visualize_graph_diff(
    sample: CLadderSample,
    edge_logits: torch.Tensor,       # (num_candidates,)  raw logits
    candidate_pairs: torch.Tensor,   # (2, num_candidates)
    threshold: float = 0.5,
    figsize: tuple[int, int] = (8, 6),
    title: str | None = None,
    node_size: int = 1800,

) -> plt.Figure:
    """
    Renders a single differential graph comparing ground truth edges against
    the model's predicted edges for a CLadderSample.

    Color coding:
      Green  solid  = TP: correctly predicted edge
      Red    solid  = FP: spurious edge (predicted, not in ground truth)
      Orange dashed = FN: missing edge  (ground truth, not predicted)

    :param sample:          CLadderSample from load_cladder_v15()
    :param edge_logits:     Raw graph_builder output for this sample (single family)
    :param candidate_pairs: Matching candidate pairs used to produce edge_logits
    :param threshold:       Sigmoid decision boundary (default 0.5)
    :param figsize:         Matplotlib figure size
    :param title:           Optional figure title override
    :return:                Matplotlib Figure
    """
    ## TODO: amend
    device = edge_logits.device
    true_edge_index = sample.data.edge_index.to(device)
    node_names = sample.node_names
    num_nodes = len(node_names)

    # Threshold logits -> predicted edge index (as of course, it won't be exactly 1.0...)
    pred_mask = torch.sigmoid(edge_logits) >= threshold
    pred_edge_index = candidate_pairs[:, pred_mask]

    # Classify
    classified = _classify_edges(true_edge_index, pred_edge_index)

    # Build DiGraph (all edges, for layout)
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for edges in classified.values():
        G.add_edges_from(edges)

    # Apply Layout (try to apply graphviz layout, otherwise spring fallback)

    try:
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw in the nodes...
    nx.draw_networkx_nodes(
        G, pos,
        node_color="#3498db",
        node_size=node_size,
        ax=ax,
    )

    # Draw the Nodes' labels
    nx.draw_networkx_labels(
        G, pos,
        labels={
            i: f"{sample.variable_keys[i]}\n{node_names[i]}"
            for i in range(num_nodes)
        },
        font_size=8,
        font_color="white",
        ax=ax,
    )

    # Draw in each edge by their class
    for cls, edges in classified.items():
        if not edges:
            continue
        style = _EDGE_STYLE[cls]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            edge_color=style["color"],
            style=style["style"],
            arrows=True,
            arrowsize=20,
            width=2.0,
            connectionstyle="arc3,rad=0.1",
            ax=ax,
            node_size=node_size,
        )

    # Add a legend
    ax.legend(
        handles=[
            Line2D([0], [0], color=s["color"], linewidth=2,
                   linestyle=s["style"], label=s["label"])
            for s in _EDGE_STYLE.values()
        ],
        loc="best",
        fontsize=9,
    )

    # Add Key Diagnostics for this graph
    n_tp = len(classified["tp"])
    n_fp = len(classified["fp"])
    n_fn = len(classified["fn"])
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else float("nan")
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else float("nan")

    default_title = (
        f"Sample {sample.sample_id}  |  {sample.graph_id}  |  "
        f"Rung {sample.rung}  |  {sample.query_type}\n"
        f"TP={n_tp}  FP={n_fp}  FN={n_fn}  "
        f"P={precision:.2f}  R={recall:.2f}"
    )
    ax.set_title(title or default_title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()

    return fig
