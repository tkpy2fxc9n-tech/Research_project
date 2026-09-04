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
python -m beamsurrogate --config runs/p0/p0_baseline/config.yaml

# 5. Smoke test (few trajectories, few epochs -- checks the pipeline runs
#    end to end and writes a filled-in metrics.json; needs at least a small
#    dataset from step 2, e.g. --n-trajectories 20)
python -m beamsurrogate --config runs/p0/p0_baseline/config.yaml --smoke-test
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
| `runs/**/figures/*.png` (879 files, 93 MB) | yes | -- |
| `runs/**/metrics.json`, `*.csv`, `*.txt`, `*.yaml`, `*.log` | yes | -- |
| `runs/**/model.pth` (94 checkpoints, 50 MB) | yes | -- |
| `runs/**/*.npz` raw prediction arrays (5.3 GB) | no | re-run the run, or `analysis/*.py` |
| `runs/**/*.gif` animations (400 MB) | no | `python analysis/p7_error_gif.py` |
| `data/*.h5` datasets (2.1 GB) | no | `dataset/make_dataset.py`, see below |

Because the checkpoints and `metrics.json` files are versioned, every
`analysis/*.py` script runs **straight after cloning**, with no training and
no dataset -- they only read `runs/*/metrics.json`. You need the datasets
only to train a model from scratch.

## How a run works

Every run is one **self-contained** `config.yaml`, living in that run's own
folder -- `runs/<phase>/<run_id>/config.yaml` -- with every field spelled out
explicitly, nothing implicit or inherited from elsewhere. Open any one file
and you see everything that run does, e.g. the top of
`runs/p2/p2_mback3/config.yaml`:

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
existing run's config, put it in a new `runs/<phase>/<new_run_id>/` folder
(the folder name becomes the run id), and change the fields that differ.
Never a copy of `src/`.**

There is no `configs/base.yaml`, no `configs/retained/`, and no `configs/`
directory at all -- every value a run uses lives in that run's own
`config.yaml`, nowhere else.

## Run folder contract

Every run writes `runs/<phase>/<run_id>/`:

```
runs/<phase>/<run_id>/
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
not source, and are regenerated by the commands above. `runs/` is only
partly gitignored: see "What is and isn't in this repository" above.

## Repository layout

```
src/beamsurrogate/    the package: physics/ (solver, waveforms, rod-network
                       junctions) · data/ (windowing, generation, split,
                       normalization) · models/ (mlp, cnn) · training/
                       (teacher_forcing, pushforward, bptt, stabilizers,
                       shared losses) · evaluate/ (rollout, metrics, plots)
                       · config.py · registry.py · cli.py
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
archives/              every earlier project directory (Beam_surrogate_model/,
                       Graph_rods_network/, Tests/, Dataset/, comparaisons/,
                       suppression_hautes_frequences/), moved here by git mv
                       once its useful code was ported into src/ --
                       read-only, kept for history, not imported by anything.
                       The former archive/ and Archives/ folders were
                       consolidated into this single archives/.
```

## Slurm

```bash
sbatch scripts/run.job runs/p0/p0_baseline/config.yaml   # one run
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
