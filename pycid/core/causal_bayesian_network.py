# Modified in this fork (modern pgmpy/pygambit support, new drawing); see FORK.md.
from __future__ import annotations

import collections
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple, Union  # noqa: F401

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference.ExactInference import BeliefPropagation

try:  # pgmpy >= 0.1.26 renamed BayesianNetwork to DiscreteBayesianNetwork
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:  # pragma: no cover - older pgmpy
    from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling

from pycid.core import drawing
from pycid.core.cpd import ConstantCPD, Outcome, ParentsNotReadyException, StochasticFunctionCPD

Relationship = Union[TabularCPD, Dict[Outcome, float], Callable[..., Union[Outcome, Dict[Outcome, float]]]]


class CausalBayesianNetwork(BayesianNetwork):
    """Causal Bayesian Network

    A Causal Bayesian Network is a Bayesian Network where the directed edges represent every causal relationship
    between the Bayesian Network's variables.
    """

    class Model(collections.UserDict):
        """
        This class keeps track of all CPDs and their domains in the form of a dictionary,
        and makes sure that whenever a CPD is added or removed, it is also added/removed from
        the BayesianNetwork list.
        """

        def __init__(self, cbn: CausalBayesianNetwork, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.cbn = cbn
            self.domain: Dict[str, List[Outcome]] = {}

        def __setitem__(self, variable: str, relationship: Relationship) -> None:
            # Update the keys
            if variable in self.keys():
                self.__delitem__(variable)
            super().__setitem__(variable, relationship)

            # Try obtaining a TabularCPD from the relationship. If it fails, do nothing
            try:
                cpd = self.to_tabular_cpd(variable, relationship)
            except ParentsNotReadyException:
                return

            # add cpd to BayesianNetwork, and update domain dictionary
            BayesianNetwork.add_cpds(self.cbn, cpd)
            old_domain = self.domain.get(variable, None)
            self.domain[variable] = cpd.state_names[variable]

            # if the domain has changed, update all descendants, and sync the state_names
            if not (old_domain and old_domain == self.domain[variable]):
                for child in self.cbn.get_children(variable):
                    if child in self.keys():
                        self.__setitem__(child, self[child])  # type: ignore

            self.sync_state_names()

        def __delitem__(self, variable: str) -> None:
            super().__delitem__(variable)
            try:
                BayesianNetwork.remove_cpds(self.cbn, variable)
            except ValueError:
                pass

        def sync_state_names(self) -> None:
            """Inform each CPD about the domains of other variables"""
            for cpd in self.cbn.get_cpds():
                cpd.store_state_names(None, None, dict(self.domain))

        def to_tabular_cpd(self, variable: str, relationship: Relationship) -> TabularCPD:
            if isinstance(relationship, TabularCPD):
                return relationship
            elif isinstance(relationship, Callable):  # type: ignore
                return StochasticFunctionCPD(variable, relationship, self.cbn)  # type: ignore
            elif isinstance(relationship, Mapping):
                return ConstantCPD(variable, relationship, self.cbn)

    def __init__(self, edges: Optional[Iterable[Tuple[str, str]]] = None, **kwargs: Any):
        """Initialize a Causal Bayesian Network

        Parameters
        ----------
        edges: A set of directed edges. Each is a pair of node labels (tail, head).
        """
        if edges is not None:
            for edge in edges:
                if not isinstance(edge, tuple) or len(edge) != 2 or not all(isinstance(node, str) for node in edge):
                    raise ValueError(f"The edge {edge} is not a tuple of two strings.")
        self.model = self.Model(self)
        super().__init__(ebunch=edges, **kwargs)

    def remove_edge(self, u: str, v: str) -> None:
        """removes an edge u to v that exists from the CBN"""
        super().remove_edge(u, v)
        if v in self.model and isinstance(self.get_cpds(v), ConstantCPD):
            self.model[v] = self.model[v]

    def add_edge(self, u: str, v: str, **kwargs: Any) -> None:
        """adds an edge from u to v to the CBN"""
        super().add_edge(u, v, **kwargs)
        if v in self.model and isinstance(self.get_cpds(v), ConstantCPD):
            self.model[v] = self.model[v]

    def add_cpds(self, *cpds: TabularCPD, **relationships: Relationship) -> None:
        """
        Add the given CPDs and initialize StochasticFunctionCPDs
        """
        for cpd in cpds:
            self.model[cpd.variable] = cpd  # type: ignore
        self.model.update(relationships)

    def remove_cpds(self, *cpds: Union[str, TabularCPD]) -> None:
        for cpd in cpds:
            del self.model[cpd.variable if isinstance(cpd, TabularCPD) else cpd]

    def is_structural_causal_model(self) -> bool:
        """Check whether self represents a Structural Causal Model (SCM).

        Nodes without parents are interpreted as exogenous, and are allowed to have any
        distribution. All other nodes are interpreted as endogenous, and are therefore
        required to have degenerate CPD tables containing only 0 and 1s.
        """
        for node in self:
            if self.get_parents(node):
                for probability in self.get_cpds(node).values.flatten():
                    if not (np.isclose(probability, 0) or np.isclose(probability, 1)):
                        return False
        return True

    def query(
        self, query: Iterable[str], context: Dict[str, Outcome], intervention: Optional[Dict[str, Outcome]] = None
    ) -> BeliefPropagation:
        """Return P(query|context, do(intervention))*P(context | do(intervention)).

        Use factor.normalize to get p(query|context, do(intervention)).
        Use context={} to get P(query).

        Parameters
        ----------
        query: A set of nodes to query.

        context: Node values to condition upon. A dictionary mapping of node => outcome.

        intervention: Interventions to apply. A dictionary mapping node => outcome.
        """
        for variable, outcome in context.items():
            if outcome not in self.model.domain[variable]:
                raise ValueError(f"The outcome {outcome} is not in the domain of {variable}")

        # Apply the intervention (if any)
        if intervention:
            cbn = self.copy()
            cbn.intervene(intervention)
        else:
            cbn = self

        # query fails if graph includes nodes not in a connected component, so we remove them
        undirected = cbn.to_undirected()
        connected_nodes: Set[str] = set().union(  # type: ignore
            *[nx.node_connected_component(undirected, var) for var in query]
        )
        context = {k: v for k, v in context.items() if k in connected_nodes}
        for node in list(cbn.nodes):
            if node not in connected_nodes:
                cbn.remove_node(node)
        if not nx.is_connected(cbn.to_undirected()):
            raise ValueError(f"query {query} contains nodes in disconnected components")

        bp = BeliefPropagation(cbn)

        with np.errstate(invalid="ignore"):  # Suppress numpy warnings for 0/0
            factor = bp.query(query, context, show_progress=False)
        return factor

    def intervene(self, intervention: Dict[str, Outcome]) -> None:
        """Given a dictionary of interventions, replace the CPDs for the relevant nodes.

        Soft interventions can be achieved by using self.add_cpds() directly.

        Parameters
        ----------
        intervention: Interventions to apply. A dictionary mapping node => value.
        """
        for variable in intervention:
            del self.model[variable]
            for p in self.get_parents(variable):  # remove ingoing edges
                self.remove_edge(p, variable)
            new_dist = {outcome: 0 for outcome in self.model.domain[variable]}
            new_dist[intervention[variable]] = 1
            self.model[variable] = new_dist

    def expected_value(
        self,
        variables: Iterable[str],
        context: Dict[str, Outcome],
        intervention: Optional[Dict[str, Outcome]] = None,
    ) -> List[float]:
        """Compute the expected value of a real-valued variable for a given context,
        under an optional intervention

        Parameters
        ----------
        variables: A set of variables to evaluate.

        context: Node values to condition upon. A dictionary mapping of node => value.

        intervention: Interventions to apply. A dictionary mapping node => value.
        """
        factor = self.query(variables, context, intervention=intervention)
        factor.normalize()  # make probs add to one

        ev = np.array([0.0 for _ in factor.variables])
        for idx, prob in np.ndenumerate(factor.values):  # type: ignore
            # idx contains the information about the value each variable takes
            # we use state_names to convert index into the actual value of the variable
            ev += prob * np.array(  # type: ignore
                [factor.state_names[variable][idx[var_idx]] for var_idx, variable in enumerate(factor.variables)]
            )
            if np.isnan(ev).any():
                raise RuntimeError("query {} | {} generated Nan from idx: {}, prob: {}, \
                                consider imputing a random decision".format(variables, context, idx, prob))
        return ev.tolist()  # type: ignore

    def sample(self, seed: Optional[int] = None) -> Dict[str, Outcome]:
        """
        Generates a sample from the joint distribution, and returns it as a dictionary {variable: outcome}
        """
        sampled_df = BayesianModelSampling(self).forward_sample(size=1, seed=seed, show_progress=False)
        return {var: sampled_df[var][0] for var in sampled_df.columns}

    def copy_without_cpds(self) -> CausalBayesianNetwork:
        """copy the CausalBayesianNetwork object"""
        new = CausalBayesianNetwork()
        new.add_nodes_from(self.nodes)
        new.add_edges_from(self.edges)
        return new

    def copy(self) -> CausalBayesianNetwork:
        """copy the MACIDBase object"""
        model_copy = self.copy_without_cpds()
        for v in self.model:
            model_copy.model[v] = self.model[v].copy() if hasattr(self.model[v], "copy") else self.model[v]
        return model_copy

    def _get_kind(self, node: str) -> str:
        """'chance' for roots and random nodes; 'deterministic' for a node with parents whose CPD is
        degenerate (or not specified yet: structure-only graphs show intermediate nodes in grey)."""
        if not self.get_parents(node):
            return "chance"
        try:
            cpd = self.get_cpds(node)
        except (ValueError, KeyError):
            cpd = None
        if cpd is None:
            return "deterministic"
        values = np.asarray(cpd.values).flatten()
        return "deterministic" if np.all(np.isclose(values, 0) | np.isclose(values, 1)) else "chance"

    def _get_color(self, node: str) -> Union[np.ndarray, str]:
        return drawing.DETERMINISTIC if self._get_kind(node) == "deterministic" else drawing.CHANCE

    def _get_shape(self, node: str) -> str:
        return "o"

    def _legend_handles(self, kinds: Set[str]) -> List[Any]:
        entries = []
        if "chance" in kinds:
            entries.append(("o", drawing.CHANCE, "chance"))
        if "deterministic" in kinds:
            entries.append(("o", drawing.DETERMINISTIC, "deterministic"))
        return drawing.legend_handles(entries)

    def _get_label(self, node: str) -> str:
        cpd = self.get_cpds(node)
        if hasattr(cpd, "label"):
            return cpd.label  # type: ignore
        elif hasattr(cpd, "__name__"):
            return cpd.__name__  # type: ignore
        else:
            return ""

    def draw(
        self,
        node_color: Optional[Callable[[str], Union[str, np.ndarray]]] = None,
        node_shape: Optional[Callable[[str], str]] = None,
        node_label: Optional[Callable[[str], str]] = None,
        layout: Union[None, str, Callable[[Any], Dict[Any, Any]], Mapping[Any, Tuple[float, float]]] = None,
        labels: Optional[Mapping[str, str]] = None,
        highlight: Optional[Iterable[Any]] = None,
        show_cpds: bool = False,
        legend: bool = True,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        show: bool = True,
        unowned: Iterable[str] = (),
    ) -> plt.Axes:
        """Draw the CBN, CID or MACID in a readable, layered style.

        Parameters
        ----------
        node_color, node_shape, node_label: optional functions node -> colour / marker / text.

        layout: "layered" (default: causes above effects, utilities at the bottom),
            "kamada_kawai", "spring", "circular", a dict node -> (x, y), or a function graph -> dict.

        labels: text to show inside nodes, e.g. {"rbar01": "$\\bar r_{01}$"}. By default names are
            typeset as math (U1 -> U_1).

        highlight: nodes, edges (u, v) or paths [n0, n1, ...] to emphasise.

        show_cpds: also print each CPD's short description under its node.

        legend: show a legend of node kinds (and agents for a MACID).

        unowned: ordinary nodes to draw as unowned utilities (e.g. a non-player stakeholder's cost).

        ax, figsize, show: matplotlib controls. Returns the Axes.
        """
        return drawing.draw_graph(
            self,
            node_color=node_color,
            node_shape=node_shape,
            node_label=node_label,
            layout=layout,
            labels=labels,
            highlight=highlight,
            show_cpds=show_cpds,
            legend=legend,
            ax=ax,
            figsize=figsize,
            show=show,
            unowned=unowned,
        )

    def draw_property(self, node_property: Callable[[str], bool], color: str = "red") -> None:
        """Draw a CBN, CID, or MACID with the nodes satisfying node_property highlighted"""

        def node_color(node: str) -> Union[np.ndarray, str]:
            if node_property(node):
                return color
            else:
                return self._get_color(node)

        self.draw(node_color=node_color)
