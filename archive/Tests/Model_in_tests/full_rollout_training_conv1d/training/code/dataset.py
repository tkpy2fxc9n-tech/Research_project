# Two ways to get a training dataset:
#   - generate_dataset_multisignal: on-the-fly, simulates a list of
#     scenarios (see scenarios.py) right now, in-process.
#   - load_hdf5_dataset: loads trajectories already simulated by
#     Dataset/Creation_dataset.py and saved to one HDF5 file -- the one
#     main.py actually uses (see its module docstring). generate_dataset_
#     multisignal/split_by_simulation are kept for standalone experiments
#     (e.g. check_equivalence.py's scenarios.sample_random_ic still needs
#     scenarios.py) but are no longer on main.py's own path.
# Either way, the train/val/test split is done by whole SIMULATION rather
# than by row -- essential since rollout evaluates entire trajectories, so a
# simulation can't be split between train and test.
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from physics import run_fd_simulation_general, run_fd_simulation_free, build_window, make_feature_columns, make_output_columns
from waves import apply_boundary

# Families load_hdf5_dataset's showcase picker looks for -- the 6 driving
# families Creation_dataset.py draws from (FAMILY_SHARES there), matching
# this project's own scenarios.FAMILY_SHARES minus "free_evolution" (an
# initial-state choice there, not a driving family in the HDF5 dataset).
SHOWCASE_FAMILIES = ["fourier", "sinusoid", "chirp", "gaussian", "shock", "filtered_random"]

# Creation_dataset.py's 3 boundary "type" options -> this project's
# dirichlet/neumann vocabulary. "Displacement" and "Velocity (integrated)"
# are both prescribed-value (dirichlet) ends -- they only differ in how the
# driving family's raw output was turned into a value before being stored,
# which is already baked into left_bc_value/right_bc_value by the time this
# project reads them.
BC_LABEL_TO_TYPE = {
    "Displacement": "dirichlet",
    "Force (du/dx)": "neumann",
    "Velocity (integrated)": "dirichlet",
}


def _n_workers_from_env() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        return max(1, int(slurm_cpus))
    return os.cpu_count() or 1


def _window_trajectory(idx, u_storage, input_fields, cfg, INPUTS, OUTPUTS) -> pd.DataFrame:
    # Windows one trajectory's full (Nt+1, Ntot) state history into one row
    # per (time step, node) -- shared by _simulate_one (on-the-fly FD
    # simulation) and load_hdf5_dataset (pre-generated trajectories), so the
    # two data sources produce byte-for-byte the same df column layout.
    nodes = cfg.nodes
    n_list = list(range(cfg.M_BACK * cfg.ndt, cfg.Nt - cfg.N_FWD * cfg.ndt + 1))

    X = np.zeros((len(n_list), len(nodes), len(INPUTS)), dtype=np.float32)
    Y = np.zeros((len(n_list), len(nodes), len(OUTPUTS)), dtype=np.float32)
    for i, n in enumerate(n_list):
        m_list = [n - lag * cfg.ndt for lag in range(cfg.M_BACK)]
        X[i] = build_window(m_list, lambda m: u_storage[m], input_fields, cfg)
        for h in range(1, cfg.N_FWD + 1):
            Y[i, :, h - 1] = u_storage[n + h * cfg.ndt, nodes] - u_storage[n, nodes]

    # n_step (and sim_idx, via broadcasting) must be repeated len(nodes)
    # times: X/Y have one row per (time step, node), but n_list only has one
    # entry per time step. Without the repeat, concat below misaligns by
    # row count (pandas unions the indices instead of raising), leaving
    # sim_idx/n_step as NaN for all but the first len(n_list) rows -- which
    # silently breaks the sim_idx-based train/val/test merge downstream.
    meta = pd.DataFrame({"sim_idx": idx, "n_step": np.repeat(n_list, len(nodes))})
    return pd.concat([
        meta.reset_index(drop=True),
        pd.DataFrame(X.reshape(-1, len(INPUTS)), columns=INPUTS),
        pd.DataFrame(Y.reshape(-1, len(OUTPUTS)), columns=OUTPUTS),
    ], axis=1)


def _simulate_one(args):
    idx, bc_left, bc_right, u0, input_fields, cfg, INPUTS, OUTPUTS = args
    if u0 is None:
        u_storage = run_fd_simulation_general(bc_left, bc_right, cfg)
    else:
        u_storage = run_fd_simulation_free(bc_left, bc_right, u0, cfg)
    df_sim = _window_trajectory(idx, u_storage, input_fields, cfg, INPUTS, OUTPUTS)
    return idx, u_storage, df_sim


def generate_dataset_multisignal(input_fields, cfg, scenarios, n_workers=None):
    INPUTS = make_feature_columns(input_fields, cfg)
    OUTPUTS = make_output_columns(cfg)

    tasks = [(idx, bc_left, bc_right, u0, input_fields, cfg, INPUTS, OUTPUTS)
             for idx, (bc_left, bc_right, u0) in enumerate(scenarios)]
    n_workers = n_workers or min(len(tasks), _n_workers_from_env())

    FIELDS = {}
    dfs = []
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            for idx, u_storage, df_sim in ex.map(_simulate_one, tasks):
                FIELDS[idx] = u_storage
                dfs.append(df_sim)
    else:
        for task in tasks:
            idx, u_storage, df_sim = _simulate_one(task)
            FIELDS[idx] = u_storage
            dfs.append(df_sim)

    df = pd.concat(dfs, ignore_index=True)
    return df, FIELDS, INPUTS, OUTPUTS


def split_by_simulation(bc_pairs: list, df: pd.DataFrame, cfg):
    n_total = len(bc_pairs)
    n_val = max(1, round(0.05 * n_total))
    n_test = max(1, round(0.05 * n_total))
    n_train = n_total - n_val - n_test

    rng = np.random.default_rng(cfg.SPLIT_SEED)
    order = rng.permutation(n_total)
    idx_train = order[:n_train].tolist()
    idx_val = order[n_train:n_train + n_val].tolist()
    idx_test = order[n_train + n_val:].tolist()

    # First test-split index is "the" rollout/visualization case.
    rollout_idx = idx_test[0]

    split_df = pd.DataFrame(
        [(i, "train") for i in idx_train]
        + [(i, "val") for i in idx_val]
        + [(i, "test") for i in idx_test],
        columns=["sim_idx", "split"],
    )
    df = df.merge(split_df, on="sim_idx", how="left")

    print("Split distribution (by simulation):")
    for s, idxs in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        n = len(idxs)
        print(f"  {s:5s} : {n:>3d} simulations ({100*n/n_total:.1f} %)")

    return df, idx_train, idx_val, idx_test, rollout_idx


def compute_norm_stats(df: pd.DataFrame, INPUTS, OUTPUTS, cfg) -> pd.DataFrame:
    train_mask = df["split"] == "train"
    cols = INPUTS + OUTPUTS
    norm_stats = pd.DataFrame({
        "mean": df.loc[train_mask, cols].mean(),
        "std": df.loc[train_mask, cols].std(),
    })
    norm_stats["std"] = norm_stats["std"].replace(0, 1)
    return norm_stats


def _decode_str_array(arr) -> list[str]:
    # h5py round-trips a dataset created from a plain list of Python str as
    # either str or bytes depending on version/backend -- normalize to str
    # either way instead of assuming one.
    return [v.decode() if isinstance(v, bytes) else v for v in arr]


def _reconstruct_full_state(u_nx: np.ndarray, left_values: np.ndarray, right_values: np.ndarray,
                             left_type: str, right_type: str, cfg) -> np.ndarray:
    # u_nx: (Nt+1, Nx), the physical-node-only trajectory Creation_dataset.py
    # stores (no ghost band -- it slices its own working array to
    # [i_left:i_right] before saving). Rebuilds the full (Nt+1, Ntot) array
    # this project's physics/rollout code expects, by re-filling the ghost
    # band at every step from the ALREADY-COMPUTED boundary values -- this
    # replays waves.apply_boundary (cheap), not the leapfrog stencil itself.
    #
    # KNOWN APPROXIMATION at index i_right specifically: waves.apply_boundary
    # deliberately leaves a Neumann boundary's OWN node untouched (comment
    # there: "the interior leapfrog stencil already computes a value there"),
    # relying on cfg.nodes = arange(i_left, i_right) meaning i_left IS one of
    # the Nx stored columns (u_nx[:,0]) -- but i_right is ONE PAST the last
    # stored column (u_nx[:,-1] maps to i_right-1), so for a Neumann RIGHT
    # boundary, index i_right's true leapfrog value was never saved by
    # Creation_dataset.py (only [i_left:i_right] was kept) and can't be
    # recovered exactly from this file alone (it needs the previous step's
    # ghost-filled state too). Approximated here as a copy of its nearest
    # physical neighbor (i_right-1) -- exact for Dirichlet (apply_boundary
    # overwrites it directly), approximate for Neumann. Affects only the
    # stencil features of nodes within SS of the right edge, only for
    # Neumann-right trajectories.
    n_steps = u_nx.shape[0]
    u_full = np.zeros((n_steps, cfg.Ntot), dtype=np.float32)
    u_full[:, cfg.i_left:cfg.i_right] = u_nx
    if right_type != "dirichlet":
        u_full[:, cfg.i_right] = u_nx[:, -1]
    for n in range(n_steps):
        apply_boundary(u_full[n], "left", left_type, float(left_values[n]), cfg)
        apply_boundary(u_full[n], "right", right_type, float(right_values[n]), cfg)
    return u_full


def _pick_family_showcase(left_family, right_family, left_driven, right_driven, split_label, n_total) -> dict:
    # One representative trajectory per family actually present in the
    # dataset, preferring a held-out (val/test) example so the showcase
    # reflects generalization rather than memorized training data. Falls
    # back to train-split trajectories only for families a val/test pass
    # didn't cover.
    showcase = {}
    for allowed_splits in (("val", "test"), ("train", "val", "test")):
        for idx in range(n_total):
            if split_label[idx] not in allowed_splits:
                continue
            if left_driven[idx] and left_family[idx] in SHOWCASE_FAMILIES and left_family[idx] not in showcase:
                showcase[left_family[idx]] = idx
            if right_driven[idx] and right_family[idx] in SHOWCASE_FAMILIES and right_family[idx] not in showcase:
                showcase[right_family[idx]] = idx
        if len(showcase) == len(SHOWCASE_FAMILIES):
            break
    return showcase


def load_hdf5_dataset(input_fields, cfg, h5_path, max_trajectories=None):
    # Loads a dataset pre-generated by Dataset/Creation_dataset.py. Returns
    # the same shapes generate_dataset_multisignal/split_by_simulation
    # produce together (df, FIELDS, INPUTS, OUTPUTS, bc_pairs, idx_train,
    # idx_val, idx_test, rollout_idx) plus family_showcase_idx, so main.py's
    # downstream code (make_dataloaders, train_full_rollout, run_rollout...)
    # is unchanged regardless of data source. Each boundary end's per-step
    # value series is wrapped as a "table" BC_WAVEFORMS entry (see waves.py)
    # so bc_value(bc, t)/apply_boundary_conditions keep working unmodified.
    import h5py  # deferred: only needed by this function, not by the module

    with h5py.File(h5_path, "r") as f:
        n_total = f["u"].shape[0] if max_trajectories is None else min(max_trajectories, f["u"].shape[0])
        u = f["u"][:n_total]
        left_bc_value = f["left_bc_value"][:n_total]
        right_bc_value = f["right_bc_value"][:n_total]
        left_label = _decode_str_array(f["left_label"][:n_total])
        right_label = _decode_str_array(f["right_label"][:n_total])
        left_family = _decode_str_array(f["left_family"][:n_total])
        right_family = _decode_str_array(f["right_family"][:n_total])
        left_driven = f["left_driven"][:n_total]
        right_driven = f["right_driven"][:n_total]
        split_label = _decode_str_array(f["split"][:n_total])

    INPUTS = make_feature_columns(input_fields, cfg)
    OUTPUTS = make_output_columns(cfg)
    t_ctrl = (np.arange(cfg.Nt + 1) * cfg.dt).tolist()

    FIELDS, bc_pairs, dfs = {}, [], []
    idx_by_split = {"train": [], "val": [], "test": []}
    for idx in range(n_total):
        left_type = BC_LABEL_TO_TYPE[left_label[idx]]
        right_type = BC_LABEL_TO_TYPE[right_label[idx]]
        u_full = _reconstruct_full_state(u[idx], left_bc_value[idx], right_bc_value[idx],
                                          left_type, right_type, cfg)
        FIELDS[idx] = u_full

        bc_pairs.append((
            (left_type, "table", {"t_ctrl": t_ctrl, "values": left_bc_value[idx].tolist(),
                                   "source_label": left_label[idx]}),
            (right_type, "table", {"t_ctrl": t_ctrl, "values": right_bc_value[idx].tolist(),
                                    "source_label": right_label[idx]}),
        ))

        dfs.append(_window_trajectory(idx, u_full, input_fields, cfg, INPUTS, OUTPUTS))
        idx_by_split[split_label[idx]].append(idx)

    df = pd.concat(dfs, ignore_index=True)
    split_df = pd.DataFrame(
        [(i, s) for s, idxs in idx_by_split.items() for i in idxs],
        columns=["sim_idx", "split"],
    )
    df = df.merge(split_df, on="sim_idx", how="left")

    idx_train, idx_val, idx_test = idx_by_split["train"], idx_by_split["val"], idx_by_split["test"]
    rollout_idx = idx_test[0]
    family_showcase_idx = _pick_family_showcase(left_family, right_family, left_driven, right_driven,
                                                 split_label, n_total)

    print(f"Loaded {n_total} trajectories from {h5_path}")
    print("Split distribution (from dataset):")
    for s, idxs in idx_by_split.items():
        print(f"  {s:5s} : {len(idxs):>4d} simulations ({100*len(idxs)/n_total:.1f} %)")

    return df, FIELDS, INPUTS, OUTPUTS, bc_pairs, idx_train, idx_val, idx_test, rollout_idx, family_showcase_idx
