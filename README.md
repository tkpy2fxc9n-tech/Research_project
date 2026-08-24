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
python dataset/make_dataset.py --profile simple  --n-trajectories 2000
python dataset/make_dataset.py --profile complex --n-trajectories 2000

# 3. Sanity-check the physics before training anything
python checks/check_equivalence.py

# 4. Run one experiment
python -m beamsurrogate --config configs/runs/p0_baseline.yaml

# 5. Smoke test (few trajectories, few epochs -- checks the pipeline runs
#    end to end and writes a filled-in metrics.json; needs at least a small
#    dataset from step 2, e.g. --n-trajectories 20)
python -m beamsurrogate --config configs/runs/p0_baseline.yaml --smoke-test
```

## How a run works

Every run is one **self-contained** file under `configs/runs/` -- every
field spelled out explicitly, nothing implicit or inherited from elsewhere.
Open any one file and you see everything that run does, e.g. the top of
`configs/runs/p2_mback3.yaml`:

```yaml
run_id: p2_mback3
M_BACK: 3
...   # every other field the run uses, spelled out below
```

No `phase` field: which phase a run belongs to is encoded only in its
`run_id`/folder name (the `pN_` prefix) -- never duplicated as a separate
value that could drift out of sync with the name.

See [`GUIDE.md`](GUIDE.md) for a full walkthrough of how a config turns into
a trained model, step by step, in plain language.

`src/beamsurrogate/config.py`'s `load_config()` just reads the YAML and
applies any `--set KEY=VALUE` CLI overrides on top -- no merging, no
indirection. String fields (`model`, `regime`, `stabilizer`, `dataset`)
select a callable from `src/beamsurrogate/registry.py` -- that indirection
is what replaces copying a whole project directory per experiment variant:

| field | values | selects |
|---|---|---|
| `model` | `mlp`, `cnn` | `registry.MODELS` |
| `regime` | `teacher_forcing`, `pushforward`, `bptt` | `registry.REGIMES` |
| `stabilizer` | `none`, `noise`, `laplacian` | `registry.STABILIZERS` |
| `dataset` | `simple`, `complex` | `registry.DATASETS` (which HDF5 file under `data/`) |

**Adding a new run costs a new YAML file -- copy an existing one close to
what you want, give it a new `run_id`, and change the fields that differ.
Never a copy of `src/`.**

There is no `configs/base.yaml` or `configs/retained/` -- every value a run
uses lives in that run's own YAML, nowhere else.

## Run folder contract

Every run writes `runs/<run_id>/`:

```
runs/<run_id>/
├── config.resolved.yaml   # the config AFTER applying --set overrides (every value; the run's own YAML is already self-contained)
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
{"run_id": "...",
 "scalars": {"r2_onestep": null, "E_short": null, "t_div": null,
             "amp_loss_pct": null, "energy_drift_pct": null,
             "n_params": null, "train_time_s": null, "net_evals_per_unit_time": null,
             "net_evals_total": null, "fd_time_med_s": null, "fd_time_std_s": null,
             "nn_time_med_s": null, "nn_time_std_s": null, "fd_flops": null, "nn_flops": null,
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
`dataset/make_dataset.py` (see Quickstart) *before* launching any run --
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
configs/               runs/<run_id>.yaml (52 total, self-contained -- no
                       base.yaml, no retained/, nothing else)
analysis/              one script per phase-level comparison (p0_*.py ...
                       p9_*.py), reads runs/*/metrics.json, writes
                       figures/ -- a second, local pass over
                       already-computed numbers, never re-runs training
dataset/               make_dataset.py, make_coarse_dataset.py,
                       make_dataset_nondim.py -- generate the HDF5 datasets,
                       run once before any training, never called automatically
checks/                check_equivalence.py, check_nondim_scaling.py --
                       sanity checks to run before trusting a training run
scripts/               run.job (Slurm launcher) + eval_*/cost_*_benchmark.py
                       (one-off evaluations and benchmarks, not part of the
                       main campaign, results not in metrics.json)
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
```

`run.job` hardcodes `REPO_ROOT=/home/aph25/Code_GH` and
`PYTHON=/home/aph25/Desktop/wave_env/bin/python` near the top -- update
these to match wherever this checkout and venv actually live, and run
`pip install -e .` once inside that venv so `python -m beamsurrogate`
resolves.

## Regression check

`checks/check_equivalence.py` verifies the differentiable torch physics
(`src/beamsurrogate/training/losses.py`, used by the `bptt`/`pushforward`
regimes) reproduces the numpy reference physics
(`src/beamsurrogate/physics/`) across all 7 signal families, before you
spend compute on an actual training run:

```bash
python checks/check_equivalence.py
```
