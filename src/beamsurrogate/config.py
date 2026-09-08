# Single source of truth for every run parameter. A run is one self-contained
# YAML file under configs/runs/ -- every field this dataclass has, spelled
# out in full (see load_config() below; configs/runs/*.yaml were generated
# from this dataclass's own defaults on 2026-08-20, when the earlier
# `inherit: base` chain -- a run stating only its diff from configs/base.yaml
# -- was dropped because it made it impossible to see what a run actually
# does without also reading base.yaml/config.py). The `model`/`regime`/
# `stabilizer`/`dataset` string fields select a callable from registry.py;
# that indirection is what replaces copy-pasting a whole project directory
# per experiment variant.
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np
import torch
import yaml


@dataclass
class Config:
    # --- run identity (bookkeeping, not physics) ---------------------------
    # No `phase` field on purpose -- which phase a run belongs to is encoded
    # only in its run_id/folder name (the `pN_` prefix), never duplicated as
    # data here. Two sources of truth for the same fact drift apart; one
    # doesn't.
    run_id: str = "unnamed"

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
    features: list[str] = field(default_factory=lambda: ["U"])   # phase 2: subset of {"U","Ut","Uxx"}
    output: str = "delta_u"     # only one supported output convention so far
    model: str = "mlp"          # registry.MODELS key: "mlp" | "cnn"          (phase 14)
    regime: str = "teacher_forcing"   # registry.REGIMES key: "teacher_forcing" | "pushforward" | "bptt"  (phase 1)
    stabilizer: str = "none"    # registry.STABILIZERS key: "none" | "noise" | "laplacian" | "both"  (phase 10)
    dataset: str = "A"          # registry.DATASETS key: "A" | "B" | "C" | "D" |
                                 # "simple_coarse_r2" | "simple_coarse_r4" | "simple_nondim"
    # Only read by pushforward/bptt (teacher_forcing's val loss is already
    # the same MSE formula as its train loss, nothing to switch). "loss"
    # (default) = the regime's own train-loss formula, evaluated on
    # validation data with no gradient -- same units/procedure as
    # train_history, so the two curves become directly comparable. "rollout"
    # = evaluate_val_rollout, a full autoregressive replay in physical
    # units, selecting/early-stopping on autonomous-rollout survival instead
    # -- must now be set explicitly. Was "rollout" by default until
    # 2026-08-20: same silent-default trap as LAMBDA_PHYSICS above --
    # 49 of 52 configs/runs/*.yaml never set this field and were silently
    # using rollout-based model selection without anyone deciding that.
    # Every config that wants rollout-based selection must pin VAL_METRIC
    # itself now.
    VAL_METRIC: str = "loss"

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
    LAMBDA_PHYSICS: float = 0.0    # phase 3 (PINN residual weight); a particular case tested
                                    # deliberately in the p3 campaign, not a silent default
                                    # every other run should inherit -- every config that
                                    # wants the physics term must pin LAMBDA_PHYSICS itself
                                    # (see configs/runs/p3_pinn_*.yaml). Was 0.01 until
                                    # 2026-08-19: that default made LAMBDA_PHYSICS silently
                                    # active in any run that didn't override it, once
                                    # training/pushforward.py grew a physics term (commit
                                    # 3bfa23e) -- p2_pushforward_val_train_same and the
                                    # p4_cnn/p4_mlp/p4_cnn_ss20_c8_8 architecture-comparison
                                    # runs were all affected without anyone intending it.
    LAMBDA_ROLLOUT: float = 1.0

    # --- pushforward regime (training/pushforward.py) ----------------------
    PF_HOPS: int = 4
    LAMBDA_PF: float = 1.0
    PF_WARMUP: int = 5             # epochs over which LAMBDA_PF ramps up from 0
    N_PF_GROUPS: int = 8

    # --- stabilizers (training/stabilizers.py) -----------------------------
    NOISE_STD: float = 0.1
    SMOOTH_ALPHA: float = 0.20     # must stay < 0.25 (smoothing stability)

    # Rest-bias suppression (physics/solver.py:compute_rest_bias) -- the
    # network's output at zero input is subtracted from every rollout step so
    # the resting zone stays at 0, instead of drifting by whatever constant
    # bias the network happens to output there. Unconditional (always
    # subtracted) until phase 10b; this flag makes it a knob so its
    # contribution to rollout stability can be isolated from
    # NOISE_STD/SMOOTH_ALPHA above. Default True preserves every existing
    # run's behavior unchanged.
    BIAS_SUPPRESSION: bool = True

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


def load_config(path: str | Path, **overrides) -> Config:
    # Reads one self-contained runs/<phase>/<run_id>/config.yaml (every field
    # spelled out, nothing implicit) and applies **overrides on top (e.g.
    # from `--set KEY=VALUE` on the CLI). A new run costs a new folder --
    # copy an existing config.yaml and change the values that differ, never
    # a copy of this function or of any code.
    #
    # run_id is NEVER read from the file -- a run's identity is its folder
    # name (runs/<phase>/<run_id>/) and that alone, so it can't drift out of
    # sync with a value repeated inside the file. A config.yaml that DOES
    # set run_id is rejected outright rather than silently overridden, so
    # the mistake surfaces immediately instead of hiding a stale value.
    path = Path(path).resolve()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if "run_id" in raw:
        raise ValueError(f"{path}: run_id must not be set in the file -- a run's identity is its "
                          f"folder name ({path.parent.name}) only, never repeated inside the config.")
    raw.update(overrides)
    known = {f.name for f in fields(Config)} - {"run_id"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config field(s) in {path}: {sorted(unknown)}")
    for tuple_field in ("HIDDEN_SIZES", "CNN_CHANNELS"):
        if tuple_field in raw and isinstance(raw[tuple_field], list):
            raw[tuple_field] = tuple(raw[tuple_field])
    raw["run_id"] = path.parent.name
    return Config(**raw)
