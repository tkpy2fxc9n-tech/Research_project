# =============================================================================
#  1D WAVE EQUATION : NEURAL-NETWORK SURROGATE
#  Pipeline: data generation -> network training -> autoregressive roll-out
# =============================================================================
#  This file chains three stages (originally three notebooks):
#    01) Generate a dataset by running an explicit finite-difference solver of
#        the 1D wave equation  u_tt = v^2 u_xx  (v = sqrt(E/rho)) for many
#        forcing parameters (amplitude A, pulsation omega), then build a table
#        of local features -> displacement increment.
#    02) Train a fully-connected network to predict that increment, and check
#        it one-step on the held-out test rows.
#    03) Use the trained network as an AUTOREGRESSIVE solver (roll-out): feed it
#        its own previous outputs and compare the rolled-out field to the real
#        simulation, with several relative-error metrics.
# =============================================================================


## Importation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

all_dfs = []                       # one DataFrame per simulation, concatenated at the end

# Parameters
E=1                                # Young's modulus (non-dimensional)
rho=2                              # density  ->  wave speed v = sqrt(E/rho)
L=1                                # bar length

Nt=300                             # number of time steps
Nx=100                             # number of spatial grid points
t_end=3                            # total simulated time

ndt = 1                            # prediction horizon (in time steps) of the learned increment

N=6                                # number of values sampled per forcing parameter

nodes = np.arange(2, Nx-2)         # interior nodes kept in the dataset (margin left at both ends)

AMPLITUDES = np.linspace(0.005, 0.1, N).round(3).tolist()   # swept forcing amplitudes A
PULSATIONS = np.linspace(7, 90, N).round(1).tolist()        # swept forcing pulsations omega
n_sims = len(AMPLITUDES)*len(PULSATIONS)                    # total number of simulations (N x N)

# We compute each simulation
for idx, (A, omega) in enumerate(product(AMPLITUDES, PULSATIONS), start=1):
    v = np.sqrt(E/rho)             # wave speed
    x  = np.linspace(0, L, Nx)     # spatial grid
    dx = x[1] - x[0]               # spatial step
    dt = t_end / Nt                # time step
    CFL = v * dt / dx              # Courant number (must be <= 1 for an explicit scheme)
    assert CFL <= 1.0, f"CFL={CFL:.3f} > 1 : schéma instable"   # stability guard

    t_pulse = 2 * np.pi / omega    # one forcing period (the pulse is active only before this time)

    def u_right(t):
        # Right-boundary forcing: a single sine pulse, then clamped to 0
        if t < t_pulse:
            return A * np.sin(omega * t)
        else:
            return 0.0

    u     = np.zeros(Nx)           # displacement at current step n
    u_dot = np.zeros(Nx)           # velocity (backward finite difference in time)
    u_2   = np.zeros(Nx)           # displacement at previous step n-1

    u_storage         = np.zeros((Nt+1, Nx))   # full displacement history
    u_dot_storage     = np.zeros((Nt, Nx))     # velocity history
    u_xx_storage      = np.zeros((Nt, Nx))     # spatial-curvature (2nd derivative) history
    delta_u_storage   = np.zeros((Nt, Nx))     # declared but unused below

    u_storage[0] = u.copy()

    rows = []

    # --- Simulation loop ---
    for n in range(Nt):
        tn = n * dt                # current physical time

        u_xx = np.zeros(Nx)
        u_xx[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2    # central 2nd-order space derivative

        u_dot = np.zeros(Nx)
        u_dot[1:-1] = (u[1:-1] - u_2[1:-1]) / dt             # backward 1st-order time derivative

        u_new = np.zeros(Nx)
        # Explicit leap-frog update of the wave equation
        u_new[1:-1] = (2.0 * u[1:-1] - u_2[1:-1] + CFL**2 * (u[2:] - 2.0 * u[1:-1] + u[:-2]))
        u_new[0]  = 0              # left boundary: clamped (fixed end)
        u_new[-1] = u_right(tn)    # right boundary: forcing

        u_dot_storage[n]   = u_dot
        u_xx_storage[n]    = u_xx

        u_2 = u.copy()             # shift history:  n-1 <- n
        u   = u_new                #                 n   <- n+1
        u_storage[n+1] = u.copy()

    # --- Dataset construction (vectorised over the nodes) ---
    for n in range(2*ndt, Nt-3):
        # For each time n, build one row per interior node. Features are taken at
        # the current time and the two previous strides (t, t-dt, t-2dt), at the
        # node and its spatial neighbours; the label is the displacement increment.
        rows.append(pd.DataFrame({
            "A":       A,
            "omega":   omega,
            
            # Center node j
            "u_dot(t,j)":     u_dot_storage[n,       nodes],
            "u_dot(t-dt,j)":  u_dot_storage[n-ndt,   nodes],
            "u_dot(t-2dt,j)": u_dot_storage[n-2*ndt, nodes],
            "u_xx(t,j)":      u_xx_storage[n,        nodes],
            "u_xx(t-dt,j)":   u_xx_storage[n-ndt,    nodes],
            "u_xx(t-2dt,j)":  u_xx_storage[n-2*ndt,  nodes],

            # Left neighbour j-1
            "u_dot(t,j-1)":     u_dot_storage[n,       nodes-1],
            "u_dot(t-dt,j-1)":  u_dot_storage[n-ndt,   nodes-1],
            "u_dot(t-2dt,j-1)": u_dot_storage[n-2*ndt, nodes-1],
            "u_xx(t,j-1)":      u_xx_storage[n,        nodes-1],
            "u_xx(t-dt,j-1)":   u_xx_storage[n-ndt,    nodes-1],
            "u_xx(t-2dt,j-1)":  u_xx_storage[n-2*ndt,  nodes-1],

            # Right neighbour j+1
            "u_dot(t,j+1)":     u_dot_storage[n,       nodes+1],
            "u_dot(t-dt,j+1)":  u_dot_storage[n-ndt,   nodes+1],
            "u_dot(t-2dt,j+1)": u_dot_storage[n-2*ndt, nodes+1],
            "u_xx(t,j+1)":      u_xx_storage[n,        nodes+1],
            "u_xx(t-dt,j+1)":   u_xx_storage[n-ndt,    nodes+1],
            "u_xx(t-2dt,j+1)":  u_xx_storage[n-2*ndt,  nodes+1],
            "u_xx(t,j-2)":      u_xx_storage[n,        nodes-2],   # wider stencil (curvature only)
            "u_xx(t,j+2)":      u_xx_storage[n,        nodes+2],

            # Output: displacement change over ndt steps (the quantity to learn)
            "delta_u@ndt":      u_storage[n+ndt, nodes] - u_storage[n, nodes],
        }))

    all_dfs.append(pd.concat(rows, ignore_index=True))

# We concatenate each simulation in the df which is the full dataset table
df = pd.concat(all_dfs, ignore_index=True)
print(df.head(0))                  # print the column header only (quick sanity check)

rng = np.random.default_rng(seed=42)   # fixed seed -> reproducible split

combos     = list(product(AMPLITUDES, PULSATIONS))   # all (A, omega) pairs
combos_arr = rng.permutation(combos)                 # shuffle the pairs

n_train = int(0.60 * n_sims)       # 60% of the simulations for training
n_val   = int(0.20 * n_sims)       # 20% for validation (the remainder -> test)

# The split is done PER SIMULATION (not per row): all rows of a given (A, omega)
# stay in the same set, which prevents leakage between train / val / test.
train_combos = set(map(tuple, combos_arr[:n_train]))
val_combos   = set(map(tuple, combos_arr[n_train:n_train + n_val]))
test_combos  = set(map(tuple, combos_arr[n_train + n_val:]))

def assign_split(row):
    # Map a row to its split via its (A, omega) key
    key = (row["A"], row["omega"])
    if key in train_combos: return "train"
    if key in val_combos:   return "val"
    return "test"

df["split"] = df.apply(assign_split, axis=1)

print("Distribution du split :")
for s in ["train", "val", "test"]:
    n = (df["split"] == s).sum()
    print(f"  {s:5s} : {n:>8,} lignes  ({100*n/len(df):.1f} %)")


# The 20 input feature names and the single output name
INPUTS = ["u_dot(t,j)","u_dot(t-dt,j)","u_dot(t-2dt,j)","u_xx(t,j)","u_xx(t-dt,j)","u_xx(t-2dt,j)","u_dot(t,j-1)","u_dot(t-dt,j-1)","u_dot(t-2dt,j-1)","u_xx(t,j-1)","u_xx(t-dt,j-1)","u_xx(t-2dt,j-1)","u_dot(t,j+1)","u_dot(t-dt,j+1)","u_dot(t-2dt,j+1)","u_xx(t,j+1)","u_xx(t-dt,j+1)","u_xx(t-2dt,j+1)","u_xx(t,j-2)","u_xx(t,j+2)"]
OUTPUTS = ["delta_u@ndt"]

# Train mask: normalization stats are computed on the TRAIN set only (avoids leakage)
train_mask = df["split"] == "train"

# Per-column mean / std, computed on the train set only
norm_stats = pd.DataFrame({
    "mean": df.loc[train_mask, INPUTS + OUTPUTS].mean(),
    "std" : df.loc[train_mask, INPUTS + OUTPUTS].std(),
})
norm_stats["std"] = norm_stats["std"].replace(0, 1)  # avoid division by zero

# Apply the standardization to the whole dataset (suffix "_n" = normalized column)
for col in INPUTS + OUTPUTS:
    df[col + "_n"] = (df[col] - norm_stats.loc[col, "mean"]) / norm_stats.loc[col, "std"]

print("Statistiques de normalisation (calculées sur train) :")
print(norm_stats.round(6))
COLS_ORDER = (
    INPUTS                                   # raw inputs
    + [c + "_n" for c in INPUTS]             # normalized inputs
    + OUTPUTS                                # raw outputs
    + [c + "_n" for c in OUTPUTS]            # normalized outputs
    + ["split"]
)
df = df[COLS_ORDER]                # reorder the columns

# Save the dataset
df.to_csv("wave_beam_dataset.csv", index=False)
print(f"Dataset sauvegardé : wave_beam_dataset.csv")
print(f"  {len(df):,} lignes × {len(df.columns)} colonnes")

# Normalization stats, to be reloaded in files 2 and 3
norm_stats.to_csv("norm_stats.csv")
print(f"Stats de normalisation sauvegardées : norm_stats.csv")

import json

# Persist the simulation/config parameters so stages 02 and 03 reuse the exact same values
params_simulation = {
    "L"          : L,
    "Nx"         : Nx,
    "Nt"         : Nt,
    "t_end"      : t_end,
    "E": E,
    "rho":rho,
    "AMPLITUDES" : AMPLITUDES,
    "PULSATIONS" : PULSATIONS,
    "n_sims"     : n_sims,
    "INPUTS":INPUTS,
    "OUTPUTS":OUTPUTS,
}

with open("params_simulation.json", "w") as f:
    json.dump(params_simulation, f, indent=2)

print("Paramètres sauvegardés : params_simulation.json")


# =============================================================================
#  STAGE 02 : reload parameters, build the network and train it
# =============================================================================

import numpy as np
import json

# Reload the parameters saved by stage 01
with open("params_simulation.json") as f:
    p = json.load(f)

L, Nx, Nt,t_end  = p["L"], p["Nx"], p["Nt"], p["t_end"]
AMPLITUDES    = p["AMPLITUDES"]
PULSATIONS    = p["PULSATIONS"]
n_sims        = p["n_sims"]
E   = p["E"]
rho = p["rho"]
v   = np.sqrt(E / rho)             # wave speed (recomputed here)
INPUTS=p["INPUTS"]
OUTPUTS=p["OUTPUTS"]

# Files produced by notebook 01
DATASET_FILE   = "wave_beam_dataset.csv"
NORMSTATS_FILE = "norm_stats.csv"

# Network architecture
HIDDEN_SIZES = [100, 70, 30]    # number of neurons per hidden layer

# Training hyper-parameters
LEARNING_RATE = 1e-3
N_EPOCHS      = 15
BATCH_SIZE    = 512

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load the dataset and the normalization stats
df         = pd.read_csv(DATASET_FILE)
norm_stats = pd.read_csv(NORMSTATS_FILE, index_col=0)

# Quick check / visualisation
print(f"Dataset : {len(df):,} lignes")
print(f"Splits  : {df['split'].value_counts().to_dict()}")

# Normalized column names
INPUTS_N = [f"{x}_n" for x in INPUTS]
OUTPUTS_N = [f"{x}_n" for x in OUTPUTS]

# Extract train / val using the split column of the dataset
train_mask = df["split"] == "train"
val_mask   = df["split"] == "val"

X_train = df.loc[train_mask, INPUTS_N].values.astype(np.float32)
y_train = df.loc[train_mask, OUTPUTS_N].values.astype(np.float32)

X_val   = df.loc[val_mask, INPUTS_N].values.astype(np.float32)
y_val   = df.loc[val_mask, OUTPUTS_N].values.astype(np.float32)

print(f"\nTrain : {len(X_train):,} exemples")
print(f"Val   : {len(X_val):,} exemples")

# Wrap the arrays in PyTorch tensors and build the DataLoaders (training set is shuffled)
train_loader = DataLoader(
    TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
    batch_size=BATCH_SIZE
)

# Fully-connected network: stacked (Linear + Tanh) blocks, then a linear output layer.
# French names kept as-is: Reseau = network, couches = layers, taille = size.
class Reseau(nn.Module):

    def __init__(self, n_inputs, n_outputs, hidden_sizes):
        super().__init__()

        couches = []
        taille_entree = n_inputs

        # NB: this loop iterates over the global HIDDEN_SIZES, not the hidden_sizes argument
        for taille in HIDDEN_SIZES:
            couches.append(nn.Linear(taille_entree, taille))  # linear layer
            couches.append(nn.Tanh())                              # non-linear activation
            taille_entree = taille

        couches.append(nn.Linear(taille_entree, n_outputs))  # output layer
        self.reseau = nn.Sequential(*couches)

    def forward(self, x):
        return self.reseau(x)


modele = Reseau(
    n_inputs    = len(INPUTS_N), 
    n_outputs   = len(OUTPUTS_N),   
    hidden_sizes = HIDDEN_SIZES,
)

n_params = sum(p.numel() for p in modele.parameters())   # total number of trainable parameters
print(modele)
print(f"\nNombre de paramètres : {n_params:,}")

criterion  = nn.MSELoss()          # mean-squared-error loss (on the normalized target)

optimiseur = torch.optim.Adam(modele.parameters(), lr=LEARNING_RATE)
# Halve the learning rate when the validation loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiseur, mode="min", factor=0.5, patience=10
)

historique_train = []              # train-loss history (for the learning curve)
historique_val   = []              # validation-loss history

meilleure_val = float("inf")   # initialise to the worst possible score
patience_compteur = 0          # kept for later; not used for early stopping here

# ---- Training loop ----
for epoch in range(1, N_EPOCHS + 1):

    # Training phase
    modele.train()
    perte_train = 0.0       # running training loss for this epoch
    for X_batch, y_batch in train_loader:
        
        optimiseur.zero_grad()                       # reset the gradients to zero

        prediction = modele(X_batch)                 # forward pass

        perte      = criterion(prediction, y_batch)  # batch loss

        perte.backward()                             # backpropagation

        optimiseur.step()                            # update the weights
        
        perte_train += perte.item()
    perte_train /= len(train_loader)                 # average over the batches

    # Validation phase: a single forward pass over the whole validation set
    modele.eval()
    with torch.no_grad():
        pred_val = modele(torch.tensor(X_val)).numpy()

    # Global validation loss
    perte_val = ((pred_val - y_val)**2).mean()
    scheduler.step(perte_val)                        # let the scheduler react to the val loss

    # Record both losses
    historique_train.append(perte_train)
    historique_val  .append(perte_val)

    if epoch % 2 == 0:
        print(f"Epoch {epoch:4d}/{N_EPOCHS}  —  "
            f"train: {perte_train:.4f}  |  "
            f"val: {perte_val:.4f}  ")
    
    # Whenever the validation improves, save this model (keep the best one)
    if perte_val < meilleure_val:
        meilleure_val = perte_val
        torch.save(modele.state_dict(), "model.pth")   # overwrite with the best so far

# Learning curve
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(historique_train, label="Train")
ax.plot(historique_val,   label="Validation")
ax.set_xlabel("Epoch") ; ax.set_ylabel("Erreur MSE")
ax.set_title("Courbe d'apprentissage")
ax.set_yscale("log")               # log scale to read the loss decay
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# Reload the best model found during training
modele.load_state_dict(torch.load("model.pth",weights_only=True ))
print(f"Meilleur modèle rechargé — val minimale : {meilleure_val:.6f}")


# =============================================================================
#  STAGE 02 (cont.) : one-step evaluation on the held-out TEST rows
# =============================================================================

# --- 1. Pick the test data ---
df_test=df[df["split"]=="test"].reset_index(drop=True)

X_new  = df_test[[c + "_n" for c in INPUTS]].values.astype(np.float32)   # normalized inputs
y_true = df_test[OUTPUTS].values   # shape (N, 1)  -- raw (un-normalized) targets

# --- 4. Prediction ---
modele.eval()
with torch.no_grad():
    y_pred_n = modele(torch.tensor(X_new)).numpy()   # shape (N, 1)  -- normalized prediction

# De-normalize column by column (back to physical units)
y_pred = np.zeros_like(y_pred_n)
for i, col in enumerate(OUTPUTS):
    mean_du = norm_stats.loc[col, "mean"]
    std_du  = norm_stats.loc[col, "std"]
    y_pred[:, i] = y_pred_n[:, i] * std_du + mean_du

# --- 5. Visualisation: one figure per output (predicted vs true) ---
fig, axes = plt.subplots(1, len(OUTPUTS), figsize=(6*len(OUTPUTS), 6), squeeze=False)
axes = axes.flatten()

#fig, axes = plt.subplots(1, len(OUTPUTS), figsize=(6*len(OUTPUTS), 6))



for i, (ax, col) in enumerate(zip(axes, OUTPUTS)):
    y_r = y_true[:, i]             # true values
    y_p = y_pred[:, i]             # predicted values
    
    ax.scatter(y_r, y_p, alpha=0.4, s=8)
    lim = max(abs(y_r).max(), abs(y_p).max())
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="prédiction parfaite")   # y = x reference line
    
    ax.set_xlabel(f"{col} réel")
    ax.set_ylabel(f"{col} prédit")
    
    mse = ((y_p - y_r)**2).mean()
    r2  = 1 - mse / y_r.var()      # coefficient of determination
    ax.set_title(f"{col}\nMSE={mse:.2e}  |  R²={r2:.3f}")
    ax.legend()
    ax.grid(True)

fig.suptitle("Test sur toutes les données test du dataset", fontsize=14)
plt.tight_layout()
plt.show()

# --- 6. Global metrics ---
for i, col in enumerate(OUTPUTS):
    mse = ((y_pred[:,i] - y_true[:,i])**2).mean()
    r2  = 1 - mse / y_true[:,i].var()
    print(f"{col:15s} : MSE = {mse:.4e}  |  R² = {r2:.4f}")


# =============================================================================
#  STAGE 03 : autoregressive roll-out and error analysis
# =============================================================================

    # Roll out cell  (advance in strides of ndt steps)
import time

A, omega = AMPLITUDES[2], PULSATIONS[2]   # pick one (A, omega) case to roll out
dx = L / (Nx - 1)
dt = t_end / Nt
CFL = v * dt / dx
x = np.linspace(0, L, Nx)
t_list = [0.5]                     # time(s) at which to plot the field profile

t_pulse = 2 * np.pi / omega

def u_right(t):
    # Same boundary forcing as in the simulation
    return A * np.sin(omega * t) if t < t_pulse else 0.0

def u_xx_de(uu):
    # Central second spatial derivative of an array uu
    out = np.zeros(Nx)
    out[1:-1] = (uu[2:] - 2*uu[1:-1] + uu[:-2]) / dx**2
    return out

def shift_clamp(a, dk):                       # spatial neighbour, clamped at the boundaries
    idx = np.clip(np.arange(Nx) + dk, 0, Nx - 1)
    return a[idx]

# Automatic decoding of column names:  "u_xx(t-dt,j+1)" -> ("u_xx", 1, +1)
def parse_feature(name):
    field, rest = name.replace("@", "").split("(")     # field = "u_xx" or "u_dot"
    time_lab, space_lab = rest.rstrip(")").split(",")
    k   = {"t": 0, "t-dt": 1, "t-2dt": 2}[time_lab]    # time index (in strides of ndt)
    off = {"j": 0, "j-1": -1, "j+1": +1, "j-2": -2, "j+2": +2}[space_lab]   # spatial offset
    return field, k, off

SPEC  = [parse_feature(c) for c in INPUTS]     # decode every input once, up front
valid = np.arange(3, Nx - 3)       # nodes actually updated by the network (margin at both ends)
mu = norm_stats.loc[OUTPUTS, "mean"].values    # target mean (for de-normalization)
sd = norm_stats.loc[OUTPUTS, "std"].values     # target std

# =====================================================
# 1) REAL simulation (ground-truth finite-difference solver)
# =====================================================

t0 = time.perf_counter()

U_reel = np.zeros((Nt + 1, Nx))
u, u_2 = np.zeros(Nx), np.zeros(Nx)
for n in range(Nt):
    u_new = np.zeros(Nx)
    u_new[1:-1] = 2*u[1:-1] - u_2[1:-1] + CFL**2 * (u[2:] - 2*u[1:-1] + u[:-2])
    u_new[-1] = u_right(n * dt)
    u_2, u = u.copy(), u_new
    U_reel[n + 1] = u.copy()

time_phys = time.perf_counter() - t0   # wall-clock time of the physical solver

# =====================================================
# 2) PREDICTED simulation (the network used as a solver)
# =====================================================

t0 = time.perf_counter()

U = np.zeros((Nt + 1, Nx))         # rolled-out displacement field

u_dot_dict = {}                    # velocity per time (kept for the final scatter plot)
u_xx_dict  = {}                    # curvature per time

for n in range(2*ndt, Nt - ndt + 1, ndt):

    # Build the time history (t, t-dt, t-2dt) FROM THE NETWORK'S OWN PAST OUTPUTS -> autoregressive
    udot = [(U[n-k*ndt] - U[n-k*ndt-1]) / dt for k in range(3)]   # velocities

    uxx  = [u_xx_de(U[n-k*ndt]) for k in range(3)]                # curvatures

    t_actuel = round(n * dt, 5)
    u_dot_dict[t_actuel] = udot[0]
    u_xx_dict[t_actuel]  = uxx[0]

    # Assemble the (n_valid_nodes, n_features) input matrix, normalized exactly as in training
    X = np.zeros((len(valid), len(INPUTS)), dtype=np.float32)
    
    for i, (name, (field, k, off)) in enumerate(zip(INPUTS, SPEC)):
        base = udot[k] if field == "u_dot" else uxx[k]          # velocity or curvature, at time k
        X[:, i] = (shift_clamp(base, off)[valid] - norm_stats.loc[name, "mean"]) \
                                                  / norm_stats.loc[name, "std"]   # spatial offset + normalize
    
    with torch.no_grad():
        sortie = modele(torch.tensor(X)).numpy()   # network output (normalized increment)
    deltas = sortie * sd + mu           # de-normalize back to a physical increment

    # Advance the field by ndt steps and re-impose the boundary conditions
    u_suiv = U[n].copy()
    u_suiv[valid] = U[n][valid] + deltas[:,0]      # add the predicted increment on interior nodes
    u_suiv[0]  = 0.0                               # left boundary clamped
    u_suiv[-1] = u_right((n + ndt - 1) * dt)       # right boundary forcing
    U[n + ndt] = u_suiv

time_pred = time.perf_counter() - t0   # wall-clock time of the surrogate roll-out

print("temps physique :", round(time_phys, 6))
print("temps predit   :", round(time_pred, 6))

# =====================================================
# 3) Errors (relative metrics between predicted and real fields)
# =====================================================

def l2_rel(pred, true, eps=1e-12):
    # Relative L2 norm of the error over the whole field
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps)

def smape(pred, true):
    # Symmetric mean absolute percentage error (over the non-zero true values)
    m = true != 0
    return np.mean(2*np.abs(true[m] - pred[m]) / (np.abs(true[m]) + np.abs(pred[m])))

def errormax(true, pred, eps=1e-12):
    # Maximum pointwise relative error
    return np.max(np.abs(true - pred) / (np.abs(true) + eps))

steps = np.arange(2*ndt, Nt + 1, ndt)   # time indices actually produced by the roll-out

temps_sim    = np.array([(m + 1) * dt for m in range(Nt)])                       # time axis
error_mape   = np.array([smape   (U[m+1], U_reel[m+1]) for m in range(Nt)])      # SMAPE per step
error_l2_rel = np.array([l2_rel  (U[m], U_reel[m]) for m in steps])              # relative L2 per step
error_max    = np.array([errormax(U_reel[m+1], U[m+1]) for m in range(Nt)])      # max relative per step

# =====================================================
# 4) Plots
# =====================================================

reel = {round(k*dt, 5): U_reel[k] for k in range(Nt + 1)}   # time -> real field
pred = {round(k*dt, 5): U[k]       for k in range(Nt + 1)}   # time -> predicted field

plt.figure(figsize=(9, 5))
for t in t_list:
    kr = min(reel, key=lambda k: abs(k - t))     # nearest available time in the real dict
    kp = min(pred, key=lambda k: abs(k - t))     # nearest available time in the predicted dict
    plt.plot(x, reel[kr], "r",  lw=2, label=f"reel  t={t}")
    plt.plot(x, pred[kp], "b--", lw=2, label=f"predit t={t}")
plt.xlabel("x"); plt.ylabel("u"); plt.legend(); plt.grid(True)

# One error-vs-time figure per metric
for err, lab, c in [(error_mape, "SMAPE", "green"),
                    (error_l2_rel, "L2 relative", "blue"),
                    (error_max, "Max relative", "red")]:
    plt.figure(figsize=(9, 4))
    plt.plot(temps_sim, err, color=c, lw=1.5)
    plt.xlabel("Time (s)"); plt.title(lab); plt.grid(True)

# =====================================================
# 5) Visualisation of u_xx as a function of u_dot (from the prediction)
# =====================================================

t_list_plot = [0.5, 0.8, 1, 1.5]   # times to scatter
colors = plt.cm.viridis(np.linspace(0, 1, len(t_list_plot)))

plt.figure(figsize=(6, 6))

for t, c in zip(t_list_plot, colors):
    k = min(u_dot_dict, key=lambda key: abs(key - t))   # nearest stored time
    plt.scatter(u_dot_dict[k][valid], u_xx_dict[k][valid], color=c, label=f"t={t}", s=10)

plt.xlabel("u_dot")
plt.ylabel("u_xx")
plt.legend()
plt.grid(True)
plt.show()
