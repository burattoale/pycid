import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import pycid  # noqa: E402
from pycid.core.drawing import layered_positions, pretty_label  # noqa: E402
from pycid.examples.story_cids import get_introduced_bias  # noqa: E402
from pycid.examples.story_macids import taxi_competition  # noqa: E402


def toy_macid() -> pycid.MACID:
    edges = [("a1", "n"), ("a2", "n"), ("n", "r1"), ("n", "r2"), ("r1", "U1"), ("r2", "U2"), ("n", "G"), ("G", "U0")]
    return pycid.MACID(edges, agent_decisions={1: ["a1"], 2: ["a2"]}, agent_utilities={1: ["U1"], 2: ["U2"]})


def test_pretty_label() -> None:
    assert pretty_label("U1") == "$U_{1}$"
    assert pretty_label("D_Al") == "$D_{Al}$"
    assert pretty_label("rbar01") == r"$\bar{r}_{01}$"
    assert pretty_label("Weather") == "Weather"


def test_layered_positions_utilities_at_bottom() -> None:
    macid = toy_macid()
    kinds = {n: macid._get_kind(n) for n in macid.nodes}
    pos = layered_positions(macid, kinds)
    bottom = min(y for _, y in pos.values())
    assert all(pos[u][1] == bottom for u in macid.utilities)
    assert all(pos[d][1] == 0 for d in macid.decisions)
    for u, v in macid.edges:
        assert pos[u][1] > pos[v][1]  # every edge points downwards


def test_kinds_and_colors() -> None:
    cid = get_introduced_bias()
    assert cid._get_kind("D") == "decision"
    assert cid._get_kind("U") == "utility"
    assert cid._get_kind("A") == "chance"
    macid = taxi_competition()
    assert macid._get_color("D1") != macid._get_color("D2")  # one colour per agent


@pytest.mark.parametrize("layout", [None, "layered", "kamada_kawai", "spring", "circular"])
def test_draw_layouts(layout: str) -> None:
    ax = get_introduced_bias().draw(layout=layout, show=False)
    assert ax is not None
    plt.close("all")


def test_draw_options() -> None:
    macid = toy_macid()
    ax = macid.draw(
        show=False,
        unowned=["U0"],
        highlight=[["a1", "n", "r2", "U2"], ("n", "G"), "U0"],
        labels={"G": "grid"},
        show_cpds=True,
    )
    assert ax is not None and ax.get_legend() is not None
    plt.close("all")
    pos = {n: (i, -i) for i, n in enumerate(macid.nodes)}
    macid.draw(layout=pos, show=False, legend=False)
    plt.close("all")
    with pytest.raises(ValueError):
        macid.draw(layout={"a1": (0, 0)}, show=False)


def test_draw_property_still_works(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    cid = get_introduced_bias()
    cid.draw_property(lambda node: node == "X")
    plt.close("all")
