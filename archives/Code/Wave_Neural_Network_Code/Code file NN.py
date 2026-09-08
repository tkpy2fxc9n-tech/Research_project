from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save all figures to disk — no window opens

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import time

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. PHYSICAL PARAMETERS ────────────────────────────────────────────────────
E     = 1      # Young's modulus
rho   = 2      # density
L     = 1      # rod length
Nx    = 100    # spatial nodes in the physical domain
Nt    = 300    # time steps
t_end = 3      # total simulation time

SS  = 5  # stencil half-width: each node reads SS neighbours on each side
ndt = 1  # prediction horizon — the NN maps state(t) → state(t + ndt·dt)

Ntot  = Nx + 2 * SS  # extended grid with SS ghost nodes on each side
x     = np.linspace(0, L, Nx)
dx    = L / (Nx - 1)
dt    = t_end / Nt
CFL   = np.sqrt(E / rho) * dt / dx
assert CFL <= 1.0, f"CFL = {CFL:.3f} > 1: unstable"

i_left  = SS          # index of the left physical boundary on the extended grid
i_right = Ntot - SS   # one past the last physical node (right ghost zone starts here)
nodes   = np.arange(i_left, i_right)   # Nx interior node indices

N          = 6
AMPLITUDES = np.linspace(0.005, 0.1, N).round(3).tolist()
PULSATIONS = np.linspace(1, 10,  N).round(1).tolist()

# ── 2. SIMULATION ─────────────────────────────────────────────────────────────
# Solve the 1-D wave equation (explicit central differences) for every
# (amplitude A, pulsation omega) pair.
# Left BC : u = 0 (fixed wall).
# Right BC: sinusoidal pulse for one period, then zero.
all_dfs = []

for A, omega in product(AMPLITUDES, PULSATIONS):
    t_pulse = 2 * np.pi / omega

    def u_right(t):
        return A * np.sin(omega * t) if t < t_pulse else 0.0

    u   = np.zeros(Ntot)
    u_1 = np.zeros(Ntot)

    u_storage    = np.zeros((Nt + 1, Ntot))
    u_xx_storage = np.zeros((Nt,     Ntot))
    u_storage[0] = u.copy()

    for n in range(Nt):
        u_xx_storage[n, i_left:i_right+1] = (
            u[i_left-1:i_right] - 2*u[i_left:i_right+1] + u[i_left+1:i_right+2]
        ) / dx**2

        u_new = np.zeros(Ntot)
        u_new[i_left:i_right+1] = (
            2*u[i_left:i_right+1] - u_1[i_left:i_right+1]
            + CFL**2 * (u[i_left-1:i_right] - 2*u[i_left:i_right+1] + u[i_left+1:i_right+2])
        )
        u_new[:i_left+1] = 0.0                   # left BC + left ghost nodes
        u_new[i_right:]  = u_right(n * dt + dt)  # right ghost nodes carry BC value

        u_1 = u.copy()
        u   = u_new
        u_storage[n + 1] = u.copy()

    # ── 3. DATASET CONSTRUCTION ──────────────────────────────────────────────
    # One row per (time step n, interior node j).
    # Inputs  : u_dot and u_xx at offsets k = -SS…+SS around j  (local stencil).
    # Target  : delta_u = u(n+ndt, j) − u(n, j)  [displacement over ndt steps]
    rows = []
    for n in range(ndt, Nt - ndt - 1):
        u_dot_n = (u_storage[n] - u_storage[n - ndt]) / (ndt * dt)  # backward FD velocity

        row = {"A": A, "omega": omega, "n_step": n}
        for k in range(-SS, SS + 1):
            row[f"u_dot_{k:+d}"] = u_dot_n[nodes + k]
            row[f"u_xx_{k:+d}"]  = u_xx_storage[n, nodes + k]
        row["delta_u"] = u_storage[n + ndt, nodes] - u_storage[n, nodes]
        rows.append(pd.DataFrame(row))

    all_dfs.append(pd.concat(rows, ignore_index=True))

df = pd.concat(all_dfs, ignore_index=True)

# ── 4. TRAIN / VAL / TEST SPLIT ───────────────────────────────────────────────
# Split by time-step index (not by row) to respect temporal causality:
# training steps come first, then validation, then test.
steps   = np.sort(df["n_step"].unique())
n_train = int(0.60 * len(steps))
n_val   = int(0.20 * len(steps))
train_steps = set(steps[:n_train])
val_steps   = set(steps[n_train:n_train + n_val])

def assign_split(row):
    if row["n_step"] in train_steps: return "train"
    if row["n_step"] in val_steps:   return "val"
    return "test"

df["split"] = df.apply(assign_split, axis=1)

INPUTS  = [f"u_dot_{k:+d}" for k in range(-SS, SS+1)] + [f"u_xx_{k:+d}" for k in range(-SS, SS+1)]
OUTPUTS = ["delta_u"]

# ── 5. NORMALISATION ─────────────────────────────────────────────────────────
# Compute mean and std on the training set only, then apply to all splits.
train_mask = df["split"] == "train"
norm_stats = pd.DataFrame({
    "mean": df.loc[train_mask, INPUTS + OUTPUTS].mean(),
    "std":  df.loc[train_mask, INPUTS + OUTPUTS].std().replace(0, 1),
})
for col in INPUTS + OUTPUTS:
    df[col + "_n"] = (df[col] - norm_stats.loc[col, "mean"]) / norm_stats.loc[col, "std"]

norm_stats.to_csv(OUTPUT_DIR / "norm_stats.csv")

# ── 6. DATA LOADERS ───────────────────────────────────────────────────────────
HIDDEN_SIZES  = [256, 150, 70]  # neurons per hidden layer
LEARNING_RATE = 1e-3
N_EPOCHS      = 20
BATCH_SIZE    = 512

INPUTS_N  = [c + "_n" for c in INPUTS]
OUTPUTS_N = [c + "_n" for c in OUTPUTS]

def make_tensors(split_name):
    mask = df["split"] == split_name
    X = df.loc[mask, INPUTS_N].values.astype(np.float32)
    y = df.loc[mask, OUTPUTS_N].values.astype(np.float32)
    return torch.tensor(X), torch.tensor(y)

X_train, y_train = make_tensors("train")
X_val,   y_val   = make_tensors("val")
X_test,  y_test  = make_tensors("test")

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# ── 7. NETWORK ────────────────────────────────────────────────────────────────
# Fully-connected MLP: hidden layers of sizes HIDDEN_SIZES neurons + Tanh.
# Input  : 2·(2·SS+1) = 22 normalised features (u_dot stencil + u_xx stencil).
# Output : 1 normalised delta_u.

class Network(nn.Module):
    def __init__(self, n_in, n_out, hidden_sizes):
        super().__init__()
        layers, size = [], n_in
        for h in hidden_sizes:
            layers += [nn.Linear(size, h), nn.Tanh()]
            size = h
        layers.append(nn.Linear(size, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = Network(len(INPUTS_N), len(OUTPUTS_N), HIDDEN_SIZES)
print(model)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── 8. TRAINING ───────────────────────────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

train_losses, val_losses = [], []
best_val = float("inf")

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for X_b, y_b in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_loader)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val).numpy()
    val_loss = float(((val_pred - y_val.numpy()) ** 2).mean())
    scheduler.step(val_loss)

    train_losses.append(epoch_loss)
    val_losses.append(val_loss)

    if epoch % 2 == 0:
        print(f"Epoch {epoch:4d}/{N_EPOCHS}  train: {epoch_loss:.4f}  val: {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), OUTPUT_DIR / "model.pth")

# ── 9. LEARNING CURVE ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label="Train")
ax.plot(val_losses,   label="Validation")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss"); ax.set_yscale("log")
ax.set_title("Learning curve"); ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learning_curve.png", dpi=150, bbox_inches="tight")
plt.close()

model.load_state_dict(torch.load(OUTPUT_DIR / "model.pth", weights_only=True))
print(f"Best model reloaded — min val loss: {best_val:.6f}")

# ── 10. TEST EVALUATION ───────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    y_pred_n = model(X_test).numpy().squeeze()

mean_du = float(norm_stats.loc["delta_u", "mean"])
std_du  = float(norm_stats.loc["delta_u", "std"])
y_pred  = y_pred_n * std_du + mean_du
y_true  = df.loc[df["split"] == "test", "delta_u"].values

mse = ((y_pred - y_true) ** 2).mean()
r2  = 1 - mse / y_true.var()
print(f"Test  MSE: {mse:.4e}   R²: {r2:.4f}")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_true, y_pred, alpha=0.4, s=8)
lim = max(np.abs(y_true).max(), np.abs(y_pred).max())
ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="perfect prediction")
ax.set_xlabel("true delta_u"); ax.set_ylabel("predicted delta_u")
ax.set_title(f"Test parity — MSE={mse:.2e}  R²={r2:.3f}")
ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "test_predictions.png", dpi=150, bbox_inches="tight")
plt.close()

# ── 11. ROLLOUT ───────────────────────────────────────────────────────────────
# Compare the physics solver against the NN rollout on one unseen scenario.
A_ro     = AMPLITUDES[2]
omega_ro = PULSATIONS[2]
t_pulse  = 2 * np.pi / omega_ro

def u_right_ro(t):
    return A_ro * np.sin(omega_ro * t) if t < t_pulse else 0.0

mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)
sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)

# 11a. Physics reference
t0 = time.perf_counter()
U_reel = np.zeros((Nt + 1, Ntot))
u, u_1 = np.zeros(Ntot), np.zeros(Ntot)
for n in range(Nt):
    u_new = np.zeros(Ntot)
    u_new[i_left:i_right+1] = (2*u[i_left:i_right+1] - u_1[i_left:i_right+1]+ CFL**2 * (u[i_left-1:i_right] - 2*u[i_left:i_right+1] + u[i_left+1:i_right+2]))
    u_new[:i_left+1] = 0.0
    u_new[i_right:]  = u_right_ro((n + 1) * dt)
    u_1, u = u.copy(), u_new
    U_reel[n + 1] = u.copy()
time_phys = time.perf_counter() - t0

# 11b. NN rollout — advances ndt steps at a time
# The bias at rest is the network output when the entire field is zero.
# Subtracting it ensures the model predicts zero increment when nothing is happening.
t0 = time.perf_counter()
U_pred = np.zeros((Nt + 1, Ntot))

X_rest = (np.zeros((len(nodes), len(INPUTS)), dtype=np.float32) - mu_in) / sd_in
model.eval()
with torch.no_grad():
    bias_rest = float(model(torch.tensor(X_rest)).numpy().ravel()[0]) * std_du + mean_du

def build_features(u_cur, u_prev_ndt):
    """Assemble and normalise the stencil feature matrix (n_nodes × n_inputs)."""
    u_dot_field = (u_cur - u_prev_ndt) / (ndt * dt)
    u_xx_field  = np.zeros(Ntot)
    u_xx_field[i_left:i_right+1] = (
        u_cur[i_left-1:i_right] - 2*u_cur[i_left:i_right+1] + u_cur[i_left+1:i_right+2]
    ) / dx**2

    X = np.zeros((len(nodes), len(INPUTS)), dtype=np.float32)
    for idx, name in enumerate(INPUTS):
        base, kstr = name.rsplit("_", 1)
        k = int(kstr)
        field = u_dot_field if base == "u_dot" else u_xx_field
        X[:, idx] = field[nodes + k]
    return (X - mu_in) / sd_in

i_last = i_right - 1   # index of the rightmost physical node (driven BC)
u_cur  = U_pred[0].copy()
u_prev = U_pred[0].copy()   # state ndt steps in the past

for n in range(0, Nt, ndt):
    # Enforce ghost nodes so the stencil never reads uninitialised values
    u_cur[:i_left]   = 0.0
    u_cur[i_last+1:] = u_cur[i_last]   # mirror right boundary into ghost zone

    X_n = build_features(u_cur, u_prev)
    with torch.no_grad():
        delta_u = model(torch.tensor(X_n)).numpy().ravel() * std_du + mean_du - bias_rest

    u_new = np.zeros(Ntot)
    u_new[nodes]   = u_cur[nodes] + delta_u
    u_new[i_left]  = 0.0                          # left BC
    u_new[i_last]  = u_right_ro((n + ndt) * dt)   # right BC

    u_prev = u_cur.copy()
    u_cur  = u_new
    U_pred[n + ndt] = u_cur.copy()

time_pred = time.perf_counter() - t0
print(f"Physics: {time_phys:.4f}s  |  NN rollout: {time_pred:.4f}s")

# ── 12. GIF ───────────────────────────────────────────────────────────────────
# Two-panel animation: wave profiles (top) + pointwise absolute error (bottom).
# Only frames at multiples of ndt are shown (the NN only updates at those steps).
frames = np.arange(0, Nt + 1, ndt)

fig_anim, (axA, axB) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

line_real, = axA.plot([], [], "r",   lw=2, label="physics")
line_pred, = axA.plot([], [], "b--", lw=2, label="NN rollout")
ymax = np.abs(U_reel[:, nodes]).max() * 1.2
axA.set_xlim(0, L); axA.set_ylim(-ymax, ymax)
axA.set_ylabel("u"); axA.legend(loc="upper right"); axA.grid(True)

line_err, = axB.plot([], [], "k", lw=1.5, label="|NN − physics|")
err_max = max(np.max([np.abs(U_pred[m, nodes] - U_reel[m, nodes]).max() for m in frames]) * 1.2, 1e-9)
axB.set_xlim(0, L); axB.set_ylim(0, err_max)
axB.set_xlabel("x"); axB.set_ylabel("absolute error"); axB.legend(loc="upper right"); axB.grid(True)

title = fig_anim.suptitle("")

def update(m):
    line_real.set_data(x, U_reel[m, nodes])
    line_pred.set_data(x, U_pred[m, nodes])
    line_err.set_data(x, np.abs(U_pred[m, nodes] - U_reel[m, nodes]))
    title.set_text(f"Wave propagation — t = {m*dt:.3f}  (step {m})")
    return line_real, line_pred, line_err, title

anim = animation.FuncAnimation(fig_anim, update, frames=frames, interval=50, blit=False)
anim.save(OUTPUT_DIR / "propagation_onde.gif", writer="pillow", fps=20, dpi=110)
plt.close(fig_anim)
print(f"GIF saved → {OUTPUT_DIR / 'propagation_onde.gif'}")

# ── 13. ERROR PLOTS ───────────────────────────────────────────────────────────
def l2_rel(pred, true, eps=1e-12):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps)

def smape(pred, true):
    m = true != 0
    return np.mean(2 * np.abs(true[m] - pred[m]) / (np.abs(true[m]) + np.abs(pred[m])))

eval_steps = np.arange(ndt, Nt + 1, ndt)
t_axis     = eval_steps * dt
l2_list    = [l2_rel(U_pred[k, nodes], U_reel[k, nodes]) for k in eval_steps]
linf_list  = [np.max(np.abs(U_pred[k, nodes] - U_reel[k, nodes])) for k in eval_steps]
smape_list = [100.0 * smape(U_pred[k, nodes], U_reel[k, nodes]) for k in eval_steps]

plt.figure(figsize=(9, 5))
plt.plot(t_axis, l2_list,   "o-", ms=3, label="L2 relative error")
plt.plot(t_axis, linf_list, "s-", ms=3, label="L∞ absolute error")
plt.yscale("log")
plt.xlabel("t"); plt.ylabel("error"); plt.grid(True, which="both"); plt.legend()
plt.title("Rollout error over time")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_time.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(t_axis, smape_list, "s-", ms=3, label="sMAPE (%)")
plt.xlabel("t"); plt.ylabel("error (%)"); plt.grid(True); plt.legend()
plt.title("sMAPE of rollout over time")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "smape_time.png", dpi=150, bbox_inches="tight")
plt.close()

print("All outputs saved to", OUTPUT_DIR)