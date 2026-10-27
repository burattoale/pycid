"""Readable, paper-style drawings of CBNs, CIDs and MACIDs.

The default layout is *layered*: causes above effects, roots placed just above the nodes they
feed, utilities on the bottom row, and the order inside each layer chosen to reduce crossings.
Node kinds follow the usual influence-diagram conventions:

* chance nodes: orange circles; deterministic nodes (degenerate CPDs with parents): grey circles
* decisions: squares; utilities: diamonds, coloured by agent (blue/green for a single agent)
* information edges (into decisions): dashed
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.lines import Line2D

if TYPE_CHECKING:  # pragma: no cover
    from pycid.core.causal_bayesian_network import CausalBayesianNetwork

CHANCE = "#D9731F"
DETERMINISTIC = "#8A979E"
SINGLE_AGENT = {"decision": "#1F6F8B", "utility": "#3E8E5E"}
AGENT_COLORS = ["#1F6F8B", "#6C4F9E", "#2A9D8F", "#B23A48", "#8C6D1F", "#3B5BA5", "#C2185B", "#5D7B2F"]
EDGE = "#59656B"
HIGHLIGHT = "#C0392B"
UNOWNED = "#2F6B47"
MARKERS = {"chance": "o", "deterministic": "o", "decision": "s", "utility": "D"}

Color = Union[str, Sequence[float], np.ndarray]


def _natural_key(name: Any) -> List[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(name))]


def pretty_label(name: Hashable) -> str:
    """Typeset node names as math: U1 -> $U_{1}$, D_Al -> $D_{Al}$, rbar01 -> $\\bar{r}_{01}$."""
    s = str(name)
    m = re.fullmatch(r"([A-Za-z]+?)(bar)?_?(\d+|_[A-Za-z0-9]+)?", s)
    if not m or len(s) > 12:
        return s
    base, bar, sub = m.group(1), m.group(2), m.group(3)
    if bar and len(base) == 1:
        base = r"\bar{" + base + "}"
    elif bar:
        base = base + "bar"
    if base in {"theta", "lambda", "kappa", "omega", "beta", "gamma", "delta", "Theta", "Phi", "Xi"}:
        base = "\\" + base
    if sub:
        sub = sub.lstrip("_")
        return f"${base}_{{{sub}}}$"
    return f"${base}$" if len(base) <= 3 or base.startswith("\\") else s


def _assign_layers(graph: nx.DiGraph, kinds: Mapping[Hashable, str]) -> Dict[Hashable, int]:
    """Layer of each node: topological generation, with roots pulled down and utilities at the bottom."""
    layer = {node: i for i, generation in enumerate(nx.topological_generations(graph)) for node in generation}
    if not layer:
        return {}
    # roots sit just above the nodes they feed (e.g. renewables next to the grid-draw node)
    for node in graph:
        if graph.in_degree(node) == 0 and graph.out_degree(node) > 0 and kinds[node] != "decision":
            layer[node] = max(0, min(layer[c] for c in graph.successors(node)) - 1)
    # utilities without children go to the bottom row
    bottom = max(layer.values())
    for node in graph:
        if kinds[node] == "utility" and graph.out_degree(node) == 0:
            layer[node] = bottom
    # compact empty layers
    remap = {old: new for new, old in enumerate(sorted(set(layer.values())))}
    return {n: remap[v] for n, v in layer.items()}


def _centred(row: List[Hashable]) -> Dict[Hashable, float]:
    return {n: i - (len(row) - 1) / 2 for i, n in enumerate(row)}


def _order_rows(graph: nx.DiGraph, layer: Mapping[Hashable, int], sweeps: int) -> Dict[int, List[Hashable]]:
    """Barycentric ordering of the nodes inside each layer, to reduce edge crossings."""
    rows: Dict[int, List[Hashable]] = defaultdict(list)
    for node in sorted(graph, key=_natural_key):
        rows[layer[node]].append(node)
    order = {n: i for row in rows.values() for i, n in enumerate(row)}
    x = {n: v for row in rows.values() for n, v in _centred(row).items()}

    def barycentre(n: Hashable) -> float:
        nbrs = [m for m in list(graph.predecessors(n)) + list(graph.successors(n)) if layer[m] != layer[n]]
        return float(np.mean([x[m] for m in nbrs])) if nbrs else x[n]

    levels = sorted(rows)
    for _ in range(sweeps):
        for direction in (levels[1:], levels[-2::-1]):
            for lv in direction:
                rows[lv].sort(key=lambda n: (barycentre(n), order[n]))
                x.update(_centred(rows[lv]))
    return rows


def layered_positions(
    graph: nx.DiGraph, kinds: Mapping[Hashable, str], sweeps: int = 8
) -> Dict[Hashable, Tuple[float, float]]:
    """Top-down layered layout with barycentric ordering (a light Sugiyama layout)."""
    layer = _assign_layers(graph, kinds)
    if not layer:
        return {}
    rows = _order_rows(graph, layer, sweeps)
    width = max(len(r) for r in rows.values())
    pos = {}
    for lv, row in rows.items():
        spread = min((width - 1) / max(len(row) - 1, 1), 1.6) if len(row) > 1 else 1.0
        for i, n in enumerate(row):
            pos[n] = ((i - (len(row) - 1) / 2) * max(spread, 1.0), -float(lv))
    return pos


def _text_color(color: Color) -> str:
    try:
        r, g, b = plt.matplotlib.colors.to_rgb(color)  # type: ignore
    except ValueError:
        return "white"
    return "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.62 else "white"


Layout = Union[None, str, Callable[[Any], Dict[Any, Any]], Mapping[Any, Tuple[float, float]]]


def _positions(cbn: "CausalBayesianNetwork", layout: Layout, kinds: Mapping[Hashable, str]) -> Dict[Any, Any]:
    if layout is None or layout == "layered":
        pos = layered_positions(cbn, kinds)
    elif isinstance(layout, Mapping):
        pos = dict(layout)
        missing = set(cbn.nodes) - set(pos)
        if missing:
            raise ValueError(f"layout misses positions for {sorted(missing, key=_natural_key)}")
    elif isinstance(layout, str):
        plain = nx.DiGraph()  # drop pgmpy edge attributes (e.g. weight=None breaks spring_layout)
        plain.add_nodes_from(cbn.nodes)
        plain.add_edges_from(cbn.edges)
        algorithms: Dict[str, Callable[[nx.DiGraph], Dict[Any, Any]]] = {
            "kamada_kawai": nx.kamada_kawai_layout,
            "spring": lambda g: nx.spring_layout(g, seed=0),
            "circular": nx.circular_layout,
        }
        if layout not in algorithms:
            raise ValueError(f"unknown layout {layout!r}: use 'layered', {', '.join(map(repr, algorithms))} or a dict")
        pos = algorithms[layout](plain)
    else:
        pos = layout(cbn)
    # node markers have a fixed size in points: keep nodes at least ~1.2 units apart
    pts = np.array(list(pos.values()), dtype=float)
    if len(pts) > 1:
        dists = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        closest = dists[np.triu_indices(len(pts), 1)].min()
        if 0 < closest < 1.2:
            pos = {n: (float(x) * 1.2 / closest, float(y) * 1.2 / closest) for n, (x, y) in pos.items()}
    return pos


def _highlighted(cbn: "CausalBayesianNetwork", highlight: Optional[Iterable[Any]]) -> Tuple[Set[Any], Set[Any]]:
    """Nodes and edges to emphasise, from a mix of nodes, edges (u, v) and paths [n0, n1, ...]."""
    nodes: Set[Any] = set()
    edges: Set[Any] = set()
    for item in highlight or []:
        if isinstance(item, tuple) and len(item) == 2 and all(n in cbn.nodes for n in item) and cbn.has_edge(*item):
            edges.add(item)
        elif isinstance(item, (list, tuple)):
            nodes.update(item)
            edges.update(zip(item[:-1], item[1:]))
        else:
            nodes.add(item)
    return nodes, edges


def _passes_through_node(pos: Mapping, u: Any, v: Any, clearance: float = 0.35) -> bool:
    """True if the straight segment u -> v runs over another node."""
    a, b = np.asarray(pos[u], dtype=float), np.asarray(pos[v], dtype=float)
    ab = b - a
    length2 = float(ab @ ab)
    if length2 == 0:
        return False
    for w, p in pos.items():
        if w in (u, v):
            continue
        t = float((np.asarray(p, dtype=float) - a) @ ab) / length2
        if 0.05 < t < 0.95 and np.linalg.norm(a + t * ab - np.asarray(p, dtype=float)) < clearance:
            return True
    return False


def _draw_edges(cbn: Any, pos: Mapping, ax: plt.Axes, kinds: Mapping, sizes: Mapping, hl_edges: Set[Any]) -> None:
    """Edges: dashed into decisions, gently curved when a straight line would run over a node."""
    nodes = list(cbn.nodes)
    for u, v in cbn.edges:
        skip = _passes_through_node(pos, u, v)
        hl = (u, v) in hl_edges
        nx.draw_networkx_edges(
            cbn,
            pos,
            edgelist=[(u, v)],
            ax=ax,
            nodelist=nodes,
            node_size=[sizes[n] for n in nodes],
            node_shape="o",
            arrowsize=16,
            width=3.0 if hl else 1.4,
            edge_color=HIGHLIGHT if hl else EDGE,
            style="dashed" if kinds[v] == "decision" else "solid",
            connectionstyle="arc3,rad=0.35" if skip else "arc3,rad=0.0",
        )


def _draw_nodes(
    cbn: Any, pos: Mapping, ax: plt.Axes, shape: Callable, colour: Callable, sizes: Mapping, hl_nodes: Set[Any]
) -> None:
    nodes = list(cbn.nodes)
    for marker in sorted({shape(n) for n in nodes}):
        group = [n for n in nodes if shape(n) == marker]
        nx.draw_networkx_nodes(
            cbn,
            pos,
            nodelist=group,
            ax=ax,
            node_shape=marker,
            node_size=[sizes[n] for n in group],
            node_color=[colour(n) for n in group],
            edgecolors=[HIGHLIGHT if n in hl_nodes else "white" for n in group],
            linewidths=[3.0 if n in hl_nodes else 1.0 for n in group],
        )


def _draw_text(cbn: Any, pos: Mapping, ax: plt.Axes, text: Mapping, colour: Callable, show_cpds: bool) -> None:
    for n in cbn.nodes:
        size = 11 if len(text[n]) < 10 else 9
        ax.text(*pos[n], text[n], ha="center", va="center", fontsize=size, color=_text_color(colour(n)), zorder=5)
        cpd_label = cbn._get_label(n) if show_cpds else ""
        if cpd_label:
            ax.text(pos[n][0], pos[n][1] - 0.38, cpd_label, ha="center", va="top", fontsize=8, color=EDGE)


def _draw_legend(cbn: Any, ax: plt.Axes, unowned: Set[str]) -> None:
    handles = cbn._legend_handles({cbn._get_kind(n) for n in cbn.nodes if n not in unowned})
    if unowned:
        handles += legend_handles([("D", UNOWNED, "unowned utility")])
    if handles:
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=min(len(handles), 6),
            frameon=False,
            fontsize=9,
            handletextpad=0.4,
            columnspacing=1.2,
        )


def draw_graph(
    cbn: "CausalBayesianNetwork",
    node_color: Optional[Callable[[str], Color]] = None,
    node_shape: Optional[Callable[[str], str]] = None,
    node_label: Optional[Callable[[str], str]] = None,
    layout: Layout = None,
    labels: Optional[Mapping[str, str]] = None,
    highlight: Optional[Iterable[Any]] = None,
    show_cpds: bool = False,
    legend: bool = True,
    ax: Optional[plt.Axes] = None,
    figsize: Optional[Tuple[float, float]] = None,
    node_size: int = 1100,
    show: bool = True,
    unowned: Iterable[str] = (),
) -> Optional[plt.Axes]:
    """Draw a CBN, CID or MACID. See ``CausalBayesianNetwork.draw`` for the parameters."""
    unowned = set(unowned)
    kinds = {n: ("utility" if n in unowned else cbn._get_kind(n)) for n in cbn.nodes}
    pos = _positions(cbn, layout, kinds)
    hl_nodes, hl_edges = _highlighted(cbn, highlight)
    xs, ys = [p[0] for p in pos.values()] or [0.0], [p[1] for p in pos.values()] or [0.0]
    if ax is None:
        if figsize is None:
            figsize = (max(4.0, 1.25 * (max(xs) - min(xs)) + 2.5), max(2.6, 1.15 * (max(ys) - min(ys)) + 1.8))
        _, ax = plt.subplots(figsize=figsize)

    base_colour = node_color if node_color else cbn._get_color

    def colour(n: str) -> Color:
        return UNOWNED if (n in unowned and node_color is None) else base_colour(n)

    shape = node_shape if node_shape else (lambda n: MARKERS[kinds[n]])
    sizes = {n: node_size * (1.25 if shape(n) == "D" else 1.0) for n in cbn.nodes}
    if node_label is not None:
        text = {n: node_label(n) for n in cbn.nodes}
    else:
        text = {n: (labels or {}).get(n, pretty_label(n)) for n in cbn.nodes}

    _draw_edges(cbn, pos, ax, kinds, sizes, hl_edges)
    _draw_nodes(cbn, pos, ax, shape, colour, sizes, hl_nodes)
    _draw_text(cbn, pos, ax, text, colour, show_cpds)
    if legend:
        _draw_legend(cbn, ax, unowned)
    ax.set_axis_off()
    ax.set_xlim(min(xs) - 0.75, max(xs) + 0.75)
    ax.set_ylim(min(ys) - 0.65, max(ys) + 0.55)
    if show:
        plt.tight_layout()
        plt.show()
        return None
    return ax


def legend_handles(entries: Iterable[Tuple[str, Color, str]]) -> List[Line2D]:
    """Legend markers from (marker, colour, text) triples."""
    return [
        Line2D([], [], marker=m, linestyle="", markersize=9, markerfacecolor=c, markeredgecolor="white", label=t)
        for m, c, t in entries
    ]
