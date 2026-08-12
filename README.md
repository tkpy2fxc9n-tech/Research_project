# beamsurrogate

An autoregressive stencil MLP/CNN surrogate model for 1D elastic wave
propagation in a beam, trained and evaluated as a config-driven ablation
campaign: **a run is a YAML file, never a copy of code.**

## Quickstart

```bash
# 1. Install the package (editable, so edits to src/ take effect immediately)
pip install -e .

# 2. Generate the two canonical datasets (you run this yourself -- it can
#    take a while at full scale; start with a small --n-trajectories to try
#    the pipeline first).
python scripts/make_dataset.py --profile simple  --n-trajectories 2000
python scripts/make_dataset.py --profile complex --n-trajectories 2000

# 3. Sanity-check the physics before training anything
python scripts/check_equivalence.py

# 4. Run one experiment
python -m beamsurrogate --config configs/runs/p0_baseline.yaml

# 5. Smoke test (few trajectories, few epochs -- checks the pipeline runs
#    end to end and writes a filled-in metrics.json; needs at least a small
#    dataset from step 2, e.g. --n-trajectories 20)
python -m beamsurrogate --config configs/runs/p0_baseline.yaml --smoke-test
```

## How a run works

Every run is one file under `configs/runs/`. It states only its
**difference** from `configs/base.yaml`:

```yaml
inherit: base
run_id: p1_mback3
phase: 1
hypothesis: H2
M_BACK: 3
```

`src/beamsurrogate/config.py`'s `load_config()` merges the `inherit` chain
and applies any `--set KEY=VALUE` CLI overrides on top. String fields
(`model`, `regime`, `stabilizer`, `dataset`) select a callable from
`src/beamsurrogate/registry.py` -- that indirection is what replaces
copying a whole project directory per experiment variant:

| field | values | selects |
|---|---|---|
| `model` | `mlp`, `cnn` | `registry.MODELS` |
| `regime` | `teacher_forcing`, `pushforward`, `bptt` | `registry.REGIMES` |
| `stabilizer` | `none`, `noise`, `laplacian` | `registry.STABILIZERS` |
| `dataset` | `simple`, `complex` | `registry.DATASETS` (which HDF5 file under `data/`) |

**Adding a 28th run costs a new YAML file with `inherit: base` plus a
handful of delta fields -- never a copy of `src/`.**

From phase 3 onward, a run inherits the *winning* config of the previous
phase instead of `base` directly -- see `configs/retained/README.md`.

## Run folder contract

Every run writes `runs/<run_id>/`:

```
runs/<run_id>/
├── config.resolved.yaml   # the config AFTER merging inherit + overrides (every value, not just the delta)
├── env.json               # git sha (+dirty), dataset sha256, hostname, SLURM_JOB_ID, python/torch versions
├── metrics.json           # fixed schema -- see below
├── model.pth
├── logs/
└── figures/                # this run's own diagnostic plots (training curve, rollout error, animation, spectrum)
```

A run should be understandable and rerunnable from its folder alone.

## `metrics.json`

Same keys for every run, `null` when a metric doesn't apply -- so
`analysis/*.py` can always index the same schema across `runs/*/metrics.json`
without special-casing which run produced it. Built by
`src/beamsurrogate/evaluate/metrics.py`:

```json
{"run_id": "...", "phase": 1, "hypothesis": "H2",
 "scalars": {"r2_onestep": null, "E_short": null, "t_div": null,
             "amp_loss_pct": null, "energy_drift_pct": null,
             "n_params": null, "train_time_s": null, "net_evals_per_unit_time": null,
             "err_near_junction": null},
 "curves": {"t": [], "err_rel_mean": [], "err_max": [], "amp_max": [], "energy": []},
 "spectrum": {"k": [], "power_pred": [], "power_ref": []}}
```

## The two datasets

Both share the same HDF5 schema (see `src/beamsurrogate/data/split.py`'s
`load_hdf5_dataset` for the reader, `src/beamsurrogate/data/generate.py`
for the writer) so the rest of the pipeline never needs to know which one
it's reading:

- **`data/beam_dataset_complex.h5`** -- 3 boundary-condition types, 6 driving
  wave families, 6 initial-state types, both ends independently driven.
- **`data/beam_dataset_simple.h5`** -- left end always at rest, right end
  always a Gaussian pulse (amplitude and width each drawn from an
  interval), Dirichlet only.

**No run ever regenerates data on the fly.** Generate both with
`scripts/make_dataset.py` (see Quickstart) *before* launching any run --
this also writes a `.sha256` sidecar next to the `.h5` file, which every
run's `env.json` records.

`data/`, `runs/`, and `figures/` are gitignored -- they're machine-local
outputs, not source.

## Repository layout

```
src/beamsurrogate/    the package: physics/ (solver, waveforms, rod-network
                       junctions) · data/ (windowing, generation, split,
                       normalization) · models/ (mlp, cnn) · training/
                       (teacher_forcing, pushforward, bptt, stabilizers,
                       shared losses) · evaluate/ (rollout, metrics, plots)
                       · config.py · registry.py · cli.py
configs/               base.yaml, runs/<run_id>.yaml (27 total), retained/
analysis/              one script per hypothesis (h1_*.py ... h9_*.py),
                       reads runs/*/metrics.json, writes figures/ -- a
                       second, local pass over already-computed numbers,
                       never re-runs training
scripts/               make_dataset.py, check_equivalence.py, run.job,
                       submit_array.sbatch
archive/               every earlier project directory (Beam_surrogate_model/,
                       Graph_rods_network/, Tests/, Dataset/), moved here by
                       git mv once its useful code was ported into src/ --
                       read-only, kept for history, not imported by anything
Archives/               older, previously-archived work (untouched by this
                       refactor; not the same directory as archive/ above)
```

## Slurm

```bash
sbatch scripts/run.job configs/runs/p0_baseline.yaml   # one run
sbatch scripts/submit_array.sbatch                                # every configs/runs/*.yaml, one array task each
```

Both scripts hardcode `REPO_ROOT=/home/aph25/Code_GH` and
`PYTHON=/home/aph25/Desktop/wave_env/bin/python` near the top -- update
these to match wherever this checkout and venv actually live, and run
`pip install -e .` once inside that venv so `python -m beamsurrogate`
resolves.

## Regression check

`scripts/check_equivalence.py` verifies the differentiable torch physics
(`src/beamsurrogate/training/losses.py`, used by the `bptt`/`pushforward`
regimes) reproduces the numpy reference physics
(`src/beamsurrogate/physics/`) across all 7 signal families, before you
spend compute on an actual training run:

```bash
python scripts/check_equivalence.py
```
