# beamsurrogate

An autoregressive stencil MLP/CNN surrogate model for 1D elastic wave
propagation in a beam, trained and evaluated as a config-driven ablation
campaign: **a run is a YAML file, never a copy of code.**

## Quickstart

```bash
# 1. Create the environment and install the package (editable, so edits to
#    src/ take effect immediately). requirements.txt pins the exact versions
#    that produced everything in runs/ -- see "Environment" below.
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2. Generate the four report-level datasets (you run this yourself -- it can
#    take a while at full scale; start with a small --n-trajectories to try
#    the pipeline first).
#    The --output name matters: registry.py looks up beam_dataset_A/B/C/D.h5,
#    while the generation profiles kept their older names.
python dataset/make_dataset.py --profile simple       --n-trajectories 2000 --output data/beam_dataset_A.h5
python dataset/make_dataset.py --profile medium       --n-trajectories 2000 --output data/beam_dataset_B.h5
python dataset/make_dataset.py --profile medium_bidir --n-trajectories 2000 --output data/beam_dataset_C.h5
python dataset/make_dataset.py --profile complex      --n-trajectories 2000 --output data/beam_dataset_D.h5

# 3. Sanity-check the physics before training anything
python checks/check_equivalence.py

# 4. Run one experiment
python -m beamsurrogate --config baseline/runs/p0_baseline/config.yaml

# 5. Smoke test (few trajectories, few epochs -- checks the pipeline runs
#    end to end and writes a filled-in metrics.json; needs at least a small
#    dataset from step 2, e.g. --n-trajectories 20)
python -m beamsurrogate --config baseline/runs/p0_baseline/config.yaml --smoke-test
```

## Environment

Everything under `runs/` was produced with this exact environment:

| | version |
|---|---|
| Python | 3.13.5 |
| torch | 2.13.0 (CUDA 13.0 build, `+cu130`) |
| numpy | 2.5.1 |
| pandas | 3.0.3 |
| h5py | 3.16.0 |
| matplotlib | 3.11.0 |
| PyYAML | 6.0.3 |

Those six are the **only** runtime dependencies -- no scipy, no
scikit-learn, no tqdm, despite what may be installed system-wide on the
original machine. `pyproject.toml` declares them unpinned so
`pip install -e .` always resolves; `requirements.txt` pins the versions in
the table.

**No GPU is required.** `pip install -r requirements.txt` gives the CPU
wheel on macOS (Apple Silicon additionally has the MPS backend) and the
default CUDA wheel on Linux. The whole pipeline runs on CPU -- a GPU only
makes training faster.

## What is and isn't in this repository

Source, configs, and the *small* outputs that let you read the results
without re-running anything are versioned. The heavy, fully regenerable
artifacts are not:

| | in git | how to get it back |
|---|---|---|
| source, configs, `requirements.txt` | yes | -- |
| `**/runs/**/figures/*.png` (879 files, 93 MB) | yes | -- |
| `**/runs/**/metrics.json`, `*.csv`, `*.txt`, `*.yaml`, `*.log` | yes | -- |
| `**/runs/**/model.pth` (94 checkpoints, 50 MB) | yes | -- |
| `**/runs/**/*.npz` raw prediction arrays (5.3 GB) | no | re-run the run, or the phase's `analysis/*.py` |
| `**/runs/**/*.gif` animations (400 MB) | no | `python rod_assembly/analysis/p7_error_gif.py` |
| `data/*.h5` datasets (2.1 GB) | no | `dataset/make_dataset.py`, see below |

Because the checkpoints and `metrics.json` files are versioned, every
phase `analysis/*.py` script runs **straight after cloning**, with no training and
no dataset -- they only read each run's own results file. You need the datasets
only to train a model from scratch.

## How a run works

Every run is one **self-contained** `config.yaml`, living in that run's own
folder -- `<phase>/runs/<run_id>/config.yaml` -- with every field spelled out
explicitly, nothing implicit or inherited from elsewhere. Open any one file
and you see everything that run does, e.g. the top of
`stencil_and_features/runs/p2_mback3/config.yaml`:

```yaml
M_BACK: 3
...   # every other field the run uses, spelled out below
```

**A run's identity is its folder name and nothing else.** `run_id` is never
written inside the file -- `load_config()` rejects it outright rather than
silently overriding it, so the name can't drift out of sync with the folder.
There is no separate `configs/` tree: the config lives next to the outputs
it produced.

No `phase` field: which phase a run belongs to is encoded only in its
`run_id`/folder name (the `pN_` prefix) -- never duplicated as a separate
value that could drift out of sync with the name.

`src/beamsurrogate/cli.py` is the entry point: it loads the config, builds
the model / regime / stabilizer from `registry.py`, trains, evaluates, and
writes the run folder described below -- read it top to bottom for the full
path from a YAML file to a trained model.

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
| `dataset` | `A`, `B`, `C`, `D`, `simple_coarse_r2`, `simple_coarse_r4`, `simple_nondim` | `registry.DATASETS` (which HDF5 file under `data/`) |

**Adding a new run costs a new folder with a `config.yaml` in it -- copy an
existing run's config, put it in a new `<phase>/runs/<new_run_id>/` folder
(the folder name becomes the run id), and change the fields that differ.
Never a copy of `src/`.**

There is no `configs/base.yaml`, no `configs/retained/`, and no `configs/`
directory at all -- every value a run uses lives in that run's own
`config.yaml`, nowhere else.

## Run folder contract

Every run writes `<phase>/runs/<run_id>/`:

```
<phase>/runs/<run_id>/
├── config.yaml            # the run's own self-contained config (versioned; this IS the run definition)
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
the phase `analysis/*.py` scripts can always index the same schema across runs
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

## The datasets

All share the same HDF5 schema (see `src/beamsurrogate/data/split.py`'s
`load_hdf5_dataset` for the reader, `src/beamsurrogate/data/generate.py`
for the writer) so the rest of the pipeline never needs to know which one
it's reading. The four report-level datasets are keyed A-D in
`registry.py`, in increasing order of difficulty:

| key | file | generation profile | content |
|---|---|---|---|
| **A** | `beam_dataset_A.h5` | `simple` | Gaussian pulse only, right-driven, Dirichlet only, never pre-excited |
| **B** | `beam_dataset_B.h5` | `medium` | same topology as A, but 5 waveform families instead of gaussian only |
| **C** | `beam_dataset_C.h5` | `medium_bidir` | same as B, except driving is two-sided: half the trajectories drive both ends |
| **D** | `beam_dataset_D.h5` | `complex` | richest: 3 BC types, 6 wave families, 6 initial-state types, both ends independent |

Three derived variants keep their original names because they are not
report-level datasets:

- **`beam_dataset_simple_coarse_r2.h5`** / **`_r4.h5`** -- A subsampled 2x /
  4x in space and time (`dataset/make_coarse_dataset.py`). Not fresh
  solver runs at coarse resolution.
- **`beam_dataset_simple_nondim.h5`** -- A regenerated with `E = rho = L = 1`
  and per-sample amplitude normalization (`dataset/make_dataset_nondim.py`).

**No run ever regenerates data on the fly.** Generate both with
`dataset/make_dataset.py` (see Quickstart) *before* launching any run --
this also writes a `.sha256` sidecar next to the `.h5` file, which every
run's `env.json` records.

`data/` is gitignored entirely -- the `.h5` files are machine-local outputs,
not source, and are regenerated by the commands above. the `runs/` folders are only
partly gitignored: see "What is and isn't in this repository" above.

## Repository layout

Each phase of the campaign is one folder, named after what that phase does,
holding its own runs and its own analysis scripts side by side. Shared code
sits alongside them.

```
baseline/              thesis 4.1 -- baseline model, diagnosis of rollout divergence
training_regime/       thesis 4.2 -- teacher forcing vs pushforward vs TBPTT
stencil_and_features/  thesis 4.3 -- M_BACK / N_FWD / ndt / input-feature ablation
interaction_check/     thesis 4.3 -- weak-coupling corroboration (N_FWD x ndt)
pinn_loss/             thesis 4.4 -- physics-informed residual weight sweep
architecture/          thesis 4.5 -- MLP width/depth sweep + linear baseline
dataset_comparison/    thesis 4.6 -- generalisation across excitation regimes (A-D)
rod_assembly/          thesis 4.7 -- transfer to the 94-rod assembly
runtime_cost/          thesis 4.8 -- FLOP model and wall-clock benchmarks
solver_validation/     thesis 2.7 -- d'Alembert + energy-conservation checks
dataset_examples/      appendix D -- sample trajectories from each dataset

beyond_thesis/         work done after the report was written, not covered by it:
  non_dimensional/       the non-dimensional claim of section 1.2, actually tested
  coarse_grids/          spatial coarsening, listed as untested in section 5.2
  stabilisation/         noise injection / Laplacian smoothing, deferred in 4.2
  mlp_vs_cnn/            CNN variant, not part of the section 4.5 sweep

src/beamsurrogate/     the package: physics/ (solver, waveforms, rod-network
                       junctions) - data/ (windowing, generation, split,
                       normalization) - models/ (mlp, cnn) - training/
                       (teacher_forcing, pushforward, bptt, stabilizers,
                       shared losses) - evaluate/ (rollout, metrics, plots)
                       - config.py - registry.py - cli.py
common/                shared by every phase's analysis scripts: _common.py
                       (run lookup, metric backfill, table rendering),
                       run_registry.py (run_id -> folder), plus the
                       cross-cutting thesis_diagrams.py and
                       backfill_training_curves.py
dataset/               make_dataset.py, make_coarse_dataset.py,
                       make_dataset_nondim.py -- generate the HDF5 datasets,
                       run once before any training, never called automatically
checks/                check_equivalence.py, check_nondim_scaling.py --
                       sanity checks to run before trusting a training run
scripts/               run.job (Slurm launcher) + eval_*.py (one-off
                       evaluations, results not in metrics.json)
archives/              every earlier project directory, kept for history and
                       imported by nothing. superseded_runs/ holds run folders
                       replaced by a later re-run, excluded from the registry
                       so they are never double-counted.
```

Each phase folder is `<phase>/runs/<run_id>/` plus `<phase>/analysis/`. Run
ids keep their `pN_` prefix, which ties a run to the campaign's phase
ordering, but no code derives a path from that prefix: `common/run_registry.py`
and `common/_common.py` look a run up by id wherever it sits in the tree, so
moving a phase folder does not break anything.

## Slurm

```bash
sbatch scripts/run.job baseline/runs/p0_baseline/config.yaml   # one run
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
