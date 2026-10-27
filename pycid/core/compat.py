"""Small compatibility layer for different pgmpy versions."""

from __future__ import annotations

from typing import Any, Hashable, Iterable, Set, Union


def ancestors_of(dag: Any, nodes: Union[Hashable, Iterable[Hashable]]) -> Set[Hashable]:
    """All ancestors of `nodes`, the nodes included (pgmpy renamed _get_ancestors_of to get_ancestors)."""
    if hasattr(dag, "get_ancestors"):
        return set(dag.get_ancestors(nodes))
    return set(dag._get_ancestors_of(nodes))  # pragma: no cover - pgmpy < 0.1.26
