# CayleyPy fork — how work is done here

This file lives **only on the `plan` branch**: notes about our process must not leak into pull requests aimed at
upstream. Read it before touching anything in this clone.

## What this repo is

- A fork of [cayleypy/cayleypy](https://github.com/cayleypy/cayleypy): `origin` = `stasdiener/cayleypy`,
  `upstream` = `cayleypy/cayleypy`.
- The work here is the 16-PR series described by `docs/plans/completed/20260803-cayleypy-integration.md` (model
  contract and checkpoints, architectures, ensembles, symmetries, beam-search features, trainer).
- **Nothing goes to upstream without an explicit instruction from the user** — no PRs, no issues, no comments. Reading
  upstream (code, issues, PRs) is fine and expected. `gh repo set-default stasdiener/cayleypy` is set in this clone,
  because `gh pr create` on a fork otherwise targets upstream by default; in the web UI, check the base repository.

## Branches

| Branch | What it is |
|---|---|
| `main` | Mirror of upstream `main`. Feature branches start here, so a PR diff is exactly its own change. |
| `plan` | This file, the plan, `docs/plans/notes/*`. **No code** — process docs never belong in an upstream PR. |
| `feat/*` | One branch per PR of the series. Base = the parent feature branch, Draft until the parent is merged. |
| `base/*` | Mechanical merges used as the base of a PR with two or more parents (`base/trainer-core`, `base/demo-checkpoint`, `base/canonical-dedup`). Without them the PR diff shows the other parent's commits too. |
| `integration/all-features` | Merge of all 16 feature branches + `cayleypy/integration_test.py` + the README/`docs/api.rst` finishing touches. Verification only — no PR, never submitted. |

## Stacked-PR protocol

- One task = one PR, small and already in upstream conventions, so it can be submitted unchanged later.
- PRs touching the same file are strictly sequential, each based on the previous one: in this series
  `PR2 → PR13 → PR14 → PR15` all edit `cayleypy/algo/beam_search.py`, and `PR8 → PR9 → PR10 → PR16` edit
  `cayleypy/train/`.
- Conflicts inside the series are only "registry" conflicts and are resolved by union of lines:
  `cayleypy/__init__.py`, `cayleypy/models/__init__.py`, `docs/api.rst`, and in `cayleypy/models/models.py` — the
  fields of `ModelConfig`, its `from_dict` and `build_model`. Submitting in dependency-graph order removes them.
- Merges inside the fork are the user's call. Upstream needs **two approvals**, and **a push after an approval resets
  it** — do not rebase an approved branch without a reason.

## Upstream rules to satisfy (README.md:129-135, 162-178, 194-216)

- PR title must describe the change ("Update graphs_lib" is called out as bad).
- No new runtime dependencies — stdlib plus what `pyproject.toml` already has. Progress output is a `verbose`
  parameter + `print` (as in `beam_search.py`); no tqdm. Unmerged PR #151 pulled dependencies and stalled.
- Google-style docstrings; reST `:param x:` in files that already use it (`beam_search.py`); comments end with a
  period; **fix pylint warnings instead of disabling them**.
- Do not add graphs to `prepare_graph`.
- Every new public symbol gets an export in `cayleypy/__init__.py` (and in the subpackage `__init__.py`) **and** an
  autosummary entry in `docs/api.rst`, in the same PR — otherwise the `build-docs` job or the API reference is wrong.
- A new predictor model must demonstrably help beam search, with public weights on Kaggle (MIT).
- Tests: `*_test.py` next to the module, `device="cpu"`, slow ones behind
  `RUN_SLOW_TESTS = os.getenv("RUN_SLOW_TESTS") == "1"` + `pytest.mark.skipif`. Seed with `torch.manual_seed`; assert
  "final loss < X% of initial", never "loss ≈ 0".

## Local commands

```bash
./lint.sh                                   # black --check ./cayleypy + pylint + mypy (the "lint" CI job)
black .                                     # the separate "format-check" job runs black over the WHOLE repo,
                                            # not just ./cayleypy - scripts outside the package count too
RUN_SLOW_TESTS=1 pytest                     # what the Ubuntu/macOS jobs run
RUN_SLOW_TESTS=1 .venv39/bin/python -m pytest    # Python 3.9 - the floor of the CI matrix
PATH=.venv/bin:$PATH docs/build_docs.sh     # the "build-docs" job; sphinx runs with -W, so warnings fail it
.venv/bin/python -m pytest --doctest-modules cayleypy/models cayleypy/train   # doctests are NOT in CI: run them
```

- `.venv` = CPython 3.12 + torch 2.13 (`uv pip install -e ".[lint,test,dev,docs]"`).
  `.venv39` = CPython 3.9 + torch 2.8, ignored via `.git/info/exclude` (not `.gitignore`).
- `.ralphex/` (execution logs) is untracked and **not** gitignored — never `git add -A`, add paths explicitly.

## Python 3.9 and torch 2.8 traps

- `Optional[X]` / `Union[X, Y]`, never `X | None` — PEP 604 is 3.10+. `list[int]`, `dict[str, int]`,
  `tuple[A, B]` in annotations are fine (PEP 585 is in 3.9).
- No `match`, no `itertools.pairwise`, no `zip(..., strict=True)`.
- In a frozen dataclass, mutable defaults need `field(default_factory=...)`.
- The 3.9 job installs **torch 2.8**, not the 2.13 in `.venv` — check new torch APIs against 2.8.
- torch >= 2.6 makes `torch.load` default to `weights_only=True`. Pass it explicitly either way.

## Codebase facts that cost time

- `CayleyGraph.get_neighbors` is **generator-major**: shape `[n_gen * n_states, ...]`, block `i` is generator `i`
  applied to all states. Getting `[n_states, n_gen]` needs `reshape((n_gen, n_states)).transpose(0, 1)`, not
  `reshape((n_states, n_gen))`. Always test children scores **column by column**: comparing whole tensors reshaped the
  same wrong way hides a transposition.
- `get_unique_states` with an identity hasher (small graphs, `encoded_state_size == 1`) returns states as a single
  column that is the hash itself.
- `Predictor.__call__` batches with `torch.cat(dim=0)`; it used to be `hstack`, which silently glued 2-D outputs along
  dim 1. Multi-output models enter through `Predictor.predict_batched`; a model with its own `score_children` (e.g.
  `QVModel`) is delegated to, batching included.
- `ModelConfig` fields are flat and additive (`n_outputs`, `tokenizer_groups`, `graph_hash`, `n_heads`,
  `dim_feedforward`, `backbone_type`, `v_consistency_weight`) — no nested configs and no dict-unions, and `from_dict`
  lists them explicitly. The factory is the public `ModelConfig.build_model` in `models.py`; `models_lib.py` is data
  only (`PREDICTOR_MODELS`).
- New beam options (`use_child_scores`, `canonical_dedup`, `non_backtracking`, `lower_bound` + `prune_above`) exist
  **only** in `search_simple`; `beam_mode="advanced"` raises a clear error for each. `search_advanced` additionally
  filters states by hashes of earlier levels (scores would need reindexing), and its loop is what the unmerged
  upstream #157 rewrites.
- `MlpModel` state_dict keys and shapes must not change at `n_outputs == 1` — the Kaggle weights in
  `PREDICTOR_MODELS` depend on them. `models_lib_test.test_loads_predictor_models` is the regression guard, and it
  really downloads the weights.
- `ModelConfig.load` accepts both a bare state dict and a self-describing checkpoint
  (`{format_version, config, state_dict}`); the checkpoint stores `graph_hash`, so loading a model for a graph with
  the right name but a different definition fails loudly instead of returning nonsense.
- Known upstream bug, deliberately untouched: an empty frontier with the default `Predictor.score_children` dies in
  `StringEncoder.encode` (`torch.min` of an empty tensor, `string_encoder.py:52`). Unreachable from `search_simple`.
- `QVModel`'s V-head gets no supervision from `Trainer` (which trains `forward`, i.e. Q), so `v_consistency_weight`
  > 0 only makes sense with weights trained elsewhere. See Post-Completion in the plan.

## Docs and plans

- `docs/plans/completed/20260803-cayleypy-integration.md` — the plan of the series: tasks with their scope changes,
  the PR status table, and the deliberately deferred items (Post-Completion).
- `docs/plans/notes/` — measurements and inventories referenced from the plan: the upstream/code inventory of task 1,
  the transformer parity blocker, the demo-checkpoint recipe with its limits.
