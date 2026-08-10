# Single source of truth for every run parameter. A run is a YAML file that
# only states its DIFFERENCE from configs/base.yaml (via `inherit: base`) --
# see load_config() below. The `model`/`regime`/`stabilizer`/`dataset`
# string fields select a callable from registry.py; that indirection is what
# replaces copy-pasting a whole project directory per experiment variant.
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class Config:
    # --- run identity (bookkeeping, not physics) ---------------------------
    run_id: str = "unnamed"
    phase: int | None = None
    hypothesis: str | None = None

    # --- beam physics --------------------------------------------------
    E: float = 1.0
    rho: float = 2.0
    L: float = 1.0
    Nt: int = 500
    Nx: int = 100
    SS: int = 11
    t_end: float = 5.0

    # M_BACK past levels -> N_FWD future horizons, spaced by ndt steps.
    ndt: int = 2
    M_BACK: int = 2
    N_FWD: int = 3

    # --- boundary-condition sampling ranges (data/generate.py) ---------
    AMP_MIN: float = 0.0005
    AMP_MAX: float = 0.005
    OMEGA_MIN: float = 0.05    # angular frequency range (sinusoid/fourier/chirp)
    OMEGA_MAX: float = 0.5
    SIGMA_MIN: float = 0.05    # gaussian pulse-width range (dedicated from OMEGA:
    SIGMA_MAX: float = 0.5     # a pulse width is not an angular frequency)

    # --- what a run trains ----------------------------------------------
    features: list[str] = field(default_factory=lambda: ["U"])   # H4: subset of {"U","Ut","Uxx"}
    output: str = "delta_u"     # only one supported output convention so far
    model: str = "mlp"          # registry.MODELS key: "mlp" | "cnn"          (H7)
    regime: str = "teacher_forcing"   # registry.REGIMES key: "teacher_forcing" | "pushforward" | "bptt"  (H5)
    stabilizer: str = "none"    # registry.STABILIZERS key: "none" | "noise" | "laplacian"  (H8)
    dataset: str = "simple"     # registry.DATASETS key: "simple" | "complex"

    # --- MLP architecture -------------------------------------------------
    HIDDEN_SIZES: tuple = (512, 256, 64)

    # --- CNN architecture (kept fixed across runs -- not equalized to the
    # MLP's parameter count and not derived from SS; see plan decision #9) --
    CNN_CHANNELS: tuple = (16, 32)
    CNN_KERNEL_SIZE: int = 5

    # --- optimization ----------------------------------------------------
    LEARNING_RATE: float = 1e-3
    N_EPOCHS: int = 100
    BATCH_SIZE: int = 64
    EARLY_STOP_PATIENCE: int = 15   # 0 disables early stopping

    # --- bptt regime (training/bptt.py) -----------------------------------
    GROUP_SIZE: int = 8
    TBPTT_HOPS: int = 10
    LAMBDA_DATA: float = 1.0
    LAMBDA_PHYSICS: float = 0.01   # H6 (PINN residual weight); 0 disables the term
    LAMBDA_ROLLOUT: float = 1.0

    # --- pushforward regime (training/pushforward.py) ----------------------
    PF_HOPS: int = 4
    LAMBDA_PF: float = 1.0
    PF_WARMUP: int = 5             # epochs over which LAMBDA_PF ramps up from 0
    N_PF_GROUPS: int = 8

    # --- stabilizers (training/stabilizers.py) -----------------------------
    NOISE_STD: float = 0.1
    SMOOTH_ALPHA: float = 0.20     # must stay < 0.25 (smoothing stability)

    # --- reproducibility ----------------------------------------------------
    SEED: int = 42
    SPLIT_SEED: int = 42

    def __post_init__(self):
        self.Ntot = self.Nx + 2 * self.SS
        self.i_left = self.SS
        self.i_right = self.Ntot - self.SS
        self.nodes = np.arange(self.i_left, self.i_right)
        self.dt = self.t_end / self.Nt
        self.dx = self.L / (self.Nx - 1)
        self.CFL = self.dt / self.dx * np.sqrt(self.E / self.rho)
        if self.CFL > 1:
            print(f"WARNING: CFL={self.CFL:.3f} > 1 -- the explicit scheme is numerically "
                  f"unstable with these Nt/Nx/t_end/L. Increase Nt and/or reduce Nx.")
        receptive_field = self.N_FWD * self.ndt
        if receptive_field > self.SS:
            print(f"WARNING: N_FWD*ndt={receptive_field} > SS={self.SS} -- a rollout hop can "
                  f"reach past the stencil built at hop start (receptive-field constraint violated).")


# Fields set by __post_init__ -- derived, not part of the YAML surface, and
# not all JSON/YAML-serializable (nodes is a numpy array).
_DERIVED_FIELDS = {"Ntot", "i_left", "i_right", "nodes", "dt", "dx", "CFL"}


def config_to_dict(cfg: Config) -> dict:
    # The plain, YAML-safe view of a resolved config -- everything a user
    # could have written in a run YAML, plus the informative scalar derived
    # quantities (dt/dx/CFL), but not `nodes` (a numpy array).
    d = {f.name: getattr(cfg, f.name) for f in fields(cfg)}
    d["dt"], d["dx"], d["CFL"] = float(cfg.dt), float(cfg.dx), float(cfg.CFL)
    for k in ("HIDDEN_SIZES", "CNN_CHANNELS"):
        d[k] = list(d[k])
    return d


def set_seeds(cfg: Config) -> None:
    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)


def _find_configs_root(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if parent.name == "configs":
            return parent
    return path.parent


def _resolve_inherit(name: str, configs_root: Path) -> Path:
    for candidate in (configs_root / f"{name}.yaml", configs_root / "retained" / f"{name}.yaml"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"inherit: {name!r} not found under {configs_root} or {configs_root/'retained'}")


def _load_yaml_chain(path: Path, configs_root: Path) -> dict:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    inherit = data.pop("inherit", None)
    if inherit:
        parent_data = _load_yaml_chain(_resolve_inherit(inherit, configs_root), configs_root)
        parent_data.update(data)
        return parent_data
    return data


def load_config(path: str | Path, **overrides) -> Config:
    # Merges the `inherit: base` chain (run YAML -> base.yaml, at most one
    # level deep in practice) then applies **overrides (e.g. from `--set
    # KEY=VALUE` on the CLI) on top. A 28th run costs a new configs/runs/*.yaml
    # with `inherit: base` plus its handful of delta fields -- never a copy
    # of this function or of any code.
    path = Path(path).resolve()
    configs_root = _find_configs_root(path)
    raw = _load_yaml_chain(path, configs_root)
    raw.pop("inherit", None)
    raw.update(overrides)
    known = {f.name for f in fields(Config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config field(s) in {path}: {sorted(unknown)}")
    for tuple_field in ("HIDDEN_SIZES", "CNN_CHANNELS"):
        if tuple_field in raw and isinstance(raw[tuple_field], list):
            raw[tuple_field] = tuple(raw[tuple_field])
    return Config(**raw)
