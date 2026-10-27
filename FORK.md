# Changes in this fork

This fork of [causalincentives/pycid](https://github.com/causalincentives/pycid) (Apache License 2.0,
see `LICENSE` and `NOTICE`) was made to use pycid in teaching with current Python tooling.
The public API is unchanged; all 213 original tests and the six tutorial notebooks pass.

## Modern dependencies

| | upstream 0.8.2 | this fork |
|---|---|---|
| Python | 3.7 – 3.10 in practice | 3.10 – 3.13 (tested 3.11, 3.13) |
| pgmpy | ==0.1.17 | >=1.1 (no PyTorch needed) |
| pygambit | ==16.0.2 (no longer builds) | >=16.3 (tested 16.3.2, 16.7.0) |
| NumPy / NetworkX | < 2 / < 3.2 | 2.x / 3.x |

* `pycid/core/causal_bayesian_network.py`: pgmpy's `BayesianNetwork` is now `DiscreteBayesianNetwork`.
* `pycid/core/compat.py` (new): `ancestors_of`, replacing pgmpy's removed private `_get_ancestors_of`
  (used in `pycid/core/get_paths.py` and `pycid/random/random_macidbase.py`).
* `pycid/export/gambit.py`: ported to the pygambit 16.3+ API (`Game.add_player`, `Game.append_move`,
  `Game.append_infoset`, `Game.set_chance_probs`, `Game.add_outcome`/`set_outcome`, `NashComputationResult`,
  unique infoset labels, exact rational payoffs and probabilities, children reached by iteration).
  `MACID.get_ne`, `get_ne_in_sg` and `get_spe` work again; solver results match the original tests.
* `setup.cfg`: unpinned requirements, `python_requires >= 3.10`.
* `tests/test_export.py`: players are no longer indexed by position in pygambit 16.7.

## Readable drawings

`pycid/core/drawing.py` (new) and `draw()` in `causal_bayesian_network.py`, `macid_base.py`, `cid.py`:

* **layered layout** by default: causes above effects, root chance nodes just above the nodes they feed,
  utilities on the bottom row, barycentric ordering to reduce crossings, gentle curves for edges that
  would cross a node;
* **conventional styles**: chance nodes orange, deterministic nodes (degenerate CPDs) grey, decisions squares,
  utilities diamonds, coloured by agent (blue/green for a single agent), **dashed information edges**,
  math-typeset labels (`U1` → $U_1$, `rbar01` → $\bar r_{01}$), and a legend;
* new options: `layout` ("layered", "kamada_kawai", "spring", "circular", a dict or a function),
  `labels`, `highlight` (nodes, edges or whole paths), `unowned` (non-player stakeholders drawn as
  utilities), `show_cpds`, `legend`, `ax`, `figsize`, `show`; `draw()` now returns the matplotlib Axes;
* the old keyword arguments (`node_color`, `node_shape`, `node_label`, a layout function) and
  `draw_property` still work;
* tests in `tests/test_drawing.py`.
