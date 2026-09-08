from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # backend non-interactif, pas de fenêtre qui s'ouvre

# Importation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation   # <-- AJOUT : pour le GIF
from itertools import product

all_dfs = []

# Parameters
E=1
rho=2
L=1

Nt=300
Nx=100
SS=5
Ntot = Nx + 2*SS

nodes = np.arange(SS, Ntot-SS)  

t_end=3
dt=t_end/Nt
dx = L / (Nx - 1)
CFL = dt/dx * np.sqrt(E/rho)

#Parameters for the roll out
ndt = 5
M_BACK = 3     # niveaux temporels en entrée : t, t-ndt, ..., t-(M_BACK-1)*ndt
N_FWD  = 3     # horizons de sortie : n+ndt, n+2ndt, ..., n+N_FWD*ndt
SMOOTH_ALPHA = 0.20     # 0 = pas de lissage ; doit rester < 0.25

N=6
LAG_LABELS = {0: "t", 1: "t-dt", 2: "t-2dt"}
def jlabel(k):
    return "j" if k == 0 else f"j{k:+d}"

AMPLITUDES = np.linspace(0.005, 0.1, N).round(3).tolist()
PULSATIONS = np.linspace(1, 10, N).round(1).tolist()
n_sims = len(AMPLITUDES)*len(PULSATIONS)

# We compute each simulation
all_dfs = []

for idx, (A, omega) in enumerate(product(AMPLITUDES, PULSATIONS), start=1):

    t_pulse = 2 * np.pi / omega
    def u_right(t):
        return A * np.sin(omega * t) if t < t_pulse else 0.0

    u    = np.zeros(Ntot)
    u_1  = np.zeros(Ntot)
    u_xx = np.zeros(Ntot)

    u_storage    = np.zeros((Nt + 1, Ntot))
    u_xx_storage = np.zeros((Nt,     Ntot))
    u_storage[0] = u.copy()

    i_left  = SS
    i_right = Ntot - SS

    # --- Boucle de simulation sur la grille étendue ---
    for n in range(Nt):
        t = n * dt
        u_xx[i_left:i_right+1] = (u[i_left-1:i_right] - 2*u[i_left:i_right+1] + u[i_left+1:i_right+2]) / dx**2

        u_new = np.zeros(Ntot)
        u_new[i_left:i_right+1] = (2.0 * u[i_left:i_right+1] - u_1[i_left:i_right+1] + CFL**2 * (u[i_left-1:i_right] - 2.0 * u[i_left:i_right+1] + u[i_left+1:i_right+2]))
        u_new[:i_left+1] = 0.0              
        u_new[i_right:]  = u_right(t + dt) 

        u_xx_storage[n] = u_xx
        u_1 = u.copy()
        u   = u_new
        u_storage[n + 1] = u.copy()

    # --- Construction du dataset : 3 instants (t, t-dt, t-2dt), voisinage ±SS ---
    rows = []

    for n in range(M_BACK*ndt, Nt - N_FWD*ndt + 1):

        row = {"A": A, "omega": omega, "n_step": n}

         # ---- ENTRÉES : M_BACK niveaux en arrière ----
        for lag in range(M_BACK):
            m   = n - lag*ndt
            lab = "t" if lag == 0 else f"t-{lag}ndt"
            udot_lag = (u_storage[m] - u_storage[m - ndt]) / (ndt * dt)
            uxx_lag  = u_xx_storage[m]
            for k in range(-SS, SS + 1):
                row[f"u_dot({lab},{jlabel(k)})"] = udot_lag[nodes + k]
                row[f"u_xx({lab},{jlabel(k)})"]  = uxx_lag[nodes + k]

       # ---- SORTIES : N_FWD horizons en avant ----
        for h in range(1, N_FWD + 1):
            row[f"delta_u@{h}ndt"] = u_storage[n + h*ndt, nodes] - u_storage[n, nodes]


        rows.append(pd.DataFrame(row))

    all_dfs.append(pd.concat(rows, ignore_index=True))

df = pd.concat(all_dfs, ignore_index=True)
print(df.head(0))
print(f"{len(df):,} lignes × {df.shape[1]} colonnes")
rng = np.random.default_rng(seed=42)

n_rows = len(df)
n_train = int(0.60 * n_rows)
n_val   = int(0.20 * n_rows)
n_test  = n_rows - n_train - n_val

split_labels = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
rng.shuffle(split_labels)
df["split"] = split_labels

print("Distribution du split :")
for s in ["train", "val", "test"]:
    n = (df["split"] == s).sum()
    print(f"  {s:5s} : {n:>8,} lignes  ({100*n/len(df):.1f} %)")

OUTPUTS = [f"delta_u@{h}ndt" for h in range(1, N_FWD + 1)]
meta    = ["A", "omega", "n_step", "split"]
INPUTS  = [c for c in df.columns if c not in meta + OUTPUTS]

# Masque train pour calculer les stats uniquement sur le train set
train_mask = df["split"] == "train"

# Calcul des stats sur le train set uniquement
norm_stats = pd.DataFrame({
    "mean": df.loc[train_mask, INPUTS + OUTPUTS].mean(),
    "std" : df.loc[train_mask, INPUTS + OUTPUTS].std(),
})
norm_stats["std"] = norm_stats["std"].replace(0, 1)  

# Application à tout le dataset
for col in INPUTS + OUTPUTS:
    df[col + "_n"] = (df[col] - norm_stats.loc[col, "mean"]) / norm_stats.loc[col, "std"]

# Architecture du réseau
HIDDEN_SIZES = [64, 32, 16] 

# Entraînement
LEARNING_RATE = 1e-3
N_EPOCHS      = 4
BATCH_SIZE    = 512

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#To check and visualise
print(f"Dataset : {len(df):,} lignes")
print(f"Splits  : {df['split'].value_counts().to_dict()}")

# Colonnes normalisées
INPUTS_N = [f"{x}_n" for x in INPUTS]
OUTPUTS_N = [f"{x}_n" for x in OUTPUTS]

X_train = df.loc[df["split"] == "train", INPUTS_N].values.astype(np.float32)
y_train = df.loc[df["split"] == "train", OUTPUTS_N].values.astype(np.float32)

X_val   = df.loc[df["split"] == "val", INPUTS_N].values.astype(np.float32)
y_val   = df.loc[df["split"] == "val", OUTPUTS_N].values.astype(np.float32)

X_test  = df.loc[df["split"] == "test", INPUTS_N].values.astype(np.float32) 
y_test  = df.loc[df["split"] == "test", OUTPUTS_N].values.astype(np.float32)  

# Conversion en tenseurs PyTorch et création des DataLoaders
train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),batch_size=BATCH_SIZE)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),batch_size=BATCH_SIZE, shuffle=False)

class Reseau(nn.Module):

    def __init__(self, n_inputs, n_outputs, hidden_sizes):
        super().__init__()

        couches = []
        taille_entree = n_inputs

        for taille in hidden_sizes:
            couches.append(nn.Linear(taille_entree, taille))  # couche linéaire
            couches.append(nn.Tanh())                           # activation non-linéaire
            taille_entree = taille

        couches.append(nn.Linear(taille_entree, n_outputs))  # couche de sortie
        self.reseau = nn.Sequential(*couches)

    def forward(self, x):
        return self.reseau(x)


modele = Reseau(
    n_inputs    = len(INPUTS_N), 
    n_outputs   = len(OUTPUTS_N),   
    hidden_sizes = HIDDEN_SIZES,
)

n_params = sum(p.numel() for p in modele.parameters())
print(modele)
print(f"\nNombre de paramètres : {n_params:,}")

criterion  = nn.MSELoss() 

optimiseur = torch.optim.Adam(modele.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiseur, mode="min", factor=0.5, patience=10
)

historique_train = []
historique_val   = []

meilleure_val = float("inf")   # on initialise au pire score possible
patience_compteur = 0          # on verra ça après

for epoch in range(1, N_EPOCHS + 1):

    # Phase entraînement
    modele.train()
    perte_train = 0.0       #initialisation of training loss
    for X_batch, y_batch in train_loader:
        X_batch = X_batch + 0.05 * torch.randn_like(X_batch)    # Noise added
        optimiseur.zero_grad()                       # remet les gradients à zéro

        prediction = modele(X_batch)                 # prédiction

        perte      = criterion(prediction, y_batch)  # erreur

        perte.backward()                             # rétropropagation

        optimiseur.step()                            # mise à jour des poids
        
        perte_train += perte.item()
    perte_train /= len(train_loader)

    # Phase validation : un seul forward pass sur tout le val
    modele.eval()
    with torch.no_grad():
        pred_val = modele(torch.tensor(X_val)).numpy()

    # Loss globale 
    perte_val = ((pred_val - y_val)**2).mean()
    scheduler.step(perte_val)

    # Loss par output
    historique_train.append(perte_train)
    historique_val  .append(perte_val)

    if epoch % 2 == 0:
        print(f"Epoch {epoch:4d}/{N_EPOCHS}  —  "
            f"train: {perte_train:.4f}  |  "
            f"val: {perte_val:.4f}  ")
    
    # Si la val s'est améliorée, on sauvegarde ce modèle
    if perte_val < meilleure_val:
        meilleure_val = perte_val
        torch.save(modele.state_dict(), SCRIPT_DIR / "model.pth")   # on écrase avec le meilleur

# Courbe d'apprentissage
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(historique_train, label="Train")
ax.plot(historique_val,   label="Validation")
ax.set_xlabel("Epoch") ; ax.set_ylabel("Erreur MSE")
ax.set_title("Courbe d'apprentissage")
ax.set_yscale("log")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "courbe_apprentissage.png", dpi=150, bbox_inches="tight")
plt.close()

# On recharge le meilleur modèle trouvé pendant l'entraînement
modele.load_state_dict(torch.load(SCRIPT_DIR / "model.pth", weights_only=True))
print(f"Meilleur modèle rechargé — val minimale : {meilleure_val:.6f}")

# --- 1. Choisir des paramètres ---
df_test=df[df["split"]=="test"].reset_index(drop=True)

X_new  = df_test[[c + "_n" for c in INPUTS]].values.astype(np.float32)
y_true = df_test[OUTPUTS].values   # shape (N, 3)

# --- 4. Prédiction ---
modele.eval()
with torch.no_grad():
    y_pred_n = modele(torch.tensor(X_new)).numpy()   # shape (N, 3)

# Dénormalisation colonne par colonne
y_pred = np.zeros_like(y_pred_n)
for i, col in enumerate(OUTPUTS):
    mean_du = norm_stats.loc[col, "mean"]
    std_du  = norm_stats.loc[col, "std"]
    y_pred[:, i] = y_pred_n[:, i] * std_du + mean_du

# --- 5. Visualisation : une figure par sortie ---
fig, axes = plt.subplots(1, len(OUTPUTS), figsize=(6*len(OUTPUTS), 6), squeeze=False)
axes = axes.flatten()

#fig, axes = plt.subplots(1, len(OUTPUTS), figsize=(6*len(OUTPUTS), 6))

for i, (ax, col) in enumerate(zip(axes, OUTPUTS)):
    y_r = y_true[:, i]
    y_p = y_pred[:, i]
    
    ax.scatter(y_r, y_p, alpha=0.4, s=8)
    lim = max(abs(y_r).max(), abs(y_p).max())
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="prédiction parfaite")
    
    ax.set_xlabel(f"{col} réel")
    ax.set_ylabel(f"{col} prédit")
    
    mse = ((y_p - y_r)**2).mean()
    r2  = 1 - mse / y_r.var()
    ax.set_title(f"{col}\nMSE={mse:.2e}  |  R²={r2:.3f}")
    ax.legend()
    ax.grid(True)

fig.suptitle("Test sur toutes les données test du dataset", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "test_predictions.png", dpi=150, bbox_inches="tight")
plt.close()

# --- 6. Métriques globales ---
for i, col in enumerate(OUTPUTS):
    mse = ((y_pred[:,i] - y_true[:,i])**2).mean()
    r2  = 1 - mse / y_true[:,i].var()
    print(f"{col:15s} : MSE = {mse:.4e}  |  R² = {r2:.4f}")

# Roll out cell  (avance par paquets de 3 pas)
import time

A, omega = AMPLITUDES[2], PULSATIONS[2]
dx = L / (Nx - 1)
dt = t_end / Nt
x = np.linspace(0, L, Nx)

t_pulse = 2 * np.pi / omega     #only one period of oscillation

def u_right(t):
    if t < t_pulse:
        return A * np.sin(omega * t)
    else:
        return 0.0

i_left, i_right = SS, Ntot - SS
nodes = np.arange(i_left, i_right) 

def u_xx_etendu(u):                    # u_xx sur la grille étendue (longueur Ntot)
    u_xx = np.zeros(Ntot)
    u_xx[i_left:i_right+1] = (u[i_left-1:i_right] - 2*u[i_left:i_right+1] + u[i_left+1:i_right+2]) / dx**2
    return u_xx

# Moyennes / écarts-types des features, dans l'ordre EXACT de INPUTS
mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)   # shape (42,)
sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)    # shape (42,)

Xz = np.zeros((len(nodes), len(INPUTS)), dtype=np.float32)
Xz = (Xz - mu_in) / sd_in

mu = norm_stats.loc[OUTPUTS, "mean"].values    
sd = norm_stats.loc[OUTPUTS, "std"].values   

with torch.no_grad():
    biais_repos = (modele(torch.tensor(Xz)).numpy() * sd + mu)[0]

# =====================================================
# 1) VRAIE simulation
# =====================================================

t0 = time.perf_counter()

U_reel = np.zeros((Nt + 1, Ntot))
u, u_1 = np.zeros(Ntot), np.zeros(Ntot)
ureel_tt_dict = np.zeros((Nt, Ntot))
ureel_xx_dict = np.zeros((Nt, Ntot))

for n in range(Nt):
    u_new = np.zeros(Ntot)
    u_new[i_left:i_right+1] = (2*u[i_left:i_right+1] - u_1[i_left:i_right+1] + CFL**2 * (u[i_left-1:i_right] - 2*u[i_left:i_right+1] + u[i_left+1:i_right+2]))
    u_new[:i_left+1] = 0.0
    t = n * dt
    u_new[i_right:]  = u_right((n+1)*dt )
    u_1, u = u.copy(), u_new
    U_reel[n + 1] = u.copy()

time_phys = time.perf_counter() - t0

# Verification of wave equation with graph 

ureel_tt_dict = np.zeros((Nt, Ntot))
ureel_xx_dict = np.zeros((Nt, Ntot))

for n in range(1, Nt):
    u_prev = U_reel[n - 1]
    u_curr = U_reel[n]
    u_next = U_reel[n + 1]
    ureel_tt_dict[n, i_left:i_right+1] = (u_next[i_left:i_right+1] - 2*u_curr[i_left:i_right+1] + u_prev[i_left:i_right+1]) / dt**2
    ureel_xx_dict[n, i_left:i_right+1] = (u_curr[i_left-1:i_right] - 2*u_curr[i_left:i_right+1] + u_curr[i_left+1:i_right+2]) / dx**2

instants = [50, 100, 200]

plt.figure()
for n in instants:
    xx_vals = ureel_xx_dict[n, i_left+1:i_right]  
    tt_vals = ureel_tt_dict[n, i_left+1:i_right] 
    plt.scatter(xx_vals, tt_vals, s=10, label=f"n = {n}")

plt.xlabel("u_xx (réel)")
plt.ylabel("u_tt (réel)")
plt.legend()
plt.grid()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.title("u_tt en fonction de u_xx (real)")
plt.savefig(OUTPUT_DIR / "utt_uxx_reel.png", dpi=150, bbox_inches="tight")
plt.close()

# =====================================================
# 2) Simulation PREDITE
# =====================================================

t0 = time.perf_counter()

history_needed = M_BACK * ndt

U = np.zeros((Nt + 1, Ntot))

mu = norm_stats.loc[OUTPUTS, "mean"].values    
sd = norm_stats.loc[OUTPUTS, "std"].values   

u_tt_dict = {}
u_xx_dict = {}

uxx_0 = np.zeros(Ntot)
uxx_1 = np.zeros(Ntot)
uxx_2 = np.zeros(Ntot)

for m in range(history_needed + 1):
    U[m] = U_reel[m]

for n in range(history_needed, Nt -  N_FWD * ndt + 1, N_FWD*ndt) :

    udot = [(U[n - lag*ndt] - U[n - (lag+1)*ndt]) / (ndt*dt) for lag in range(M_BACK)]
    uxx  = [u_xx_etendu(U[n - lag*ndt])                       for lag in range(M_BACK)]


    t_actuel = round((n-history_needed) * dt, 5)
    u_tt_dict[t_actuel] = udot[0][nodes]
    u_xx_dict[t_actuel]  = uxx[0][nodes]

    X = np.zeros((len(nodes), len(INPUTS)), dtype=np.float32)
    col = 0
    for lag in range(M_BACK):
        for off in range(-SS, SS + 1):
            X[:, col] = udot[lag][nodes + off]; col += 1
            X[:, col] = uxx[lag][nodes + off];  col += 1
    X = (X - mu_in) / sd_in

    with torch.no_grad():
        sortie = modele(torch.tensor(X)).numpy()
    deltas = sortie * sd + mu - biais_repos

    for h in range(1, N_FWD + 1):
        s = n + h*ndt
        U[s, nodes]     = U[n, nodes] + deltas[:, h-1]
        U[s, :i_left+1] = 0.0
        U[s, i_right:]  = u_right(s * dt)

        if SMOOTH_ALPHA > 0:
            j0, j1 = i_left + 1, i_right            # nœuds intérieurs (hors bords)
            lap = U[s, j0-1:j1-1] - 2*U[s, j0:j1] + U[s, j0+1:j1+1]
            U[s, j0:j1] += SMOOTH_ALPHA * lap

time_pred = time.perf_counter() - t0

upred_tt_dict = np.zeros((Nt, Ntot))
upred_xx_dict = np.zeros((Nt, Ntot))

for n in range(ndt, Nt - ndt + 1, ndt):
    u_prev = U[n - ndt ]
    u_curr = U[n       ]
    u_next = U[n + ndt]
    upred_tt_dict[n, i_left:i_right+1] = (u_next[i_left:i_right+1] - 2*u_curr[i_left:i_right+1] + u_prev[i_left:i_right+1]) / (ndt*dt)**2
    upred_xx_dict[n, i_left:i_right+1] = (u_curr[i_left-1:i_right] - 2*u_curr[i_left:i_right+1] + u_curr[i_left+1:i_right+2]) / dx**2

instants = [5, 100, 150]

plt.figure()
for n in instants:
    xx_vals = upred_xx_dict[n, i_left+1:i_right]
    tt_vals = upred_tt_dict[n, i_left+1:i_right]
    plt.scatter(xx_vals, tt_vals, s=10, label=f"n = {n}")

plt.xlabel("u_xx (predit)")
plt.ylabel("u_tt (predit)")
plt.grid()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.legend()
plt.title("u_tt en fonction de u_xx (prediction)")
plt.savefig(OUTPUT_DIR / "utt_uxx_predit.png", dpi=150, bbox_inches="tight")
plt.close()

print("temps physique :", round(time_phys, 6))
print("temps predit   :", round(time_pred, 6))

m =  70 # multiple de ndt, dans la zone ou tu vois l'ecart
print("bord droit reel  :", U_reel[m, i_right:])
print("bord droit predit:", U[m, i_right-4:i_right])
print("interieur reel   :", U_reel[m, i_right-4:i_right])
print("interieur predit :", U[m + history_needed, i_right-4:i_right])

# =====================================================
# 3) Erreurs computation
# =====================================================

def l2_rel(pred, true, eps=1e-12):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps)

def smape(pred, true):
    m = true != 0
    return np.mean(2*np.abs(true[m] - pred[m]) / (np.abs(true[m]) + np.abs(pred[m])))

def errormax(true, pred, eps=1e-12):
    return np.max(np.abs(true - pred) / (np.abs(true) + eps))

steps = np.arange(2*ndt, Nt + 1, ndt)

# =====================================================
# 4) Traces
# =====================================================


reel = {round(m*dt, 5): U_reel[m] for m in range(Nt + 1)}
pred = {round(m*dt, 5): U[m] for m in range(Nt + 1)}

# (a) GIF : propagation de l'onde réelle vs prédite + erreur spatiale
# U n'est rempli qu'aux multiples de ndt pendant le rollout -> on n'anime
# que ces pas, sinon la courbe prédite retomberait à zéro entre deux blocs.
frames = np.arange(0, Nt + 1, ndt)

fig_anim, (axA, axB) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ligne_reel, = axA.plot([], [], "r",   lw=2, label="réel")
ligne_pred, = axA.plot([], [], "b--", lw=2, label="prédit")
ymax = np.abs(U_reel[:, nodes]).max() * 1.2
axA.set_xlim(0, L); axA.set_ylim(-ymax, ymax)
axA.set_ylabel("u"); axA.legend(loc="upper right"); axA.grid(True)

ligne_err, = axB.plot([], [], "k", lw=1.5, label="|prédit - réel|")
err_max = max(np.max([np.abs(U[m, nodes] - U_reel[m, nodes]).max() for m in frames]) * 1.2, 1e-9)
axB.set_xlim(0, L); axB.set_ylim(0, err_max)
axB.set_xlabel("x"); axB.set_ylabel("erreur absolue"); axB.legend(loc="upper right"); axB.grid(True)

titre = fig_anim.suptitle("")

def maj(m):
    ligne_reel.set_data(x, U_reel[m, nodes])
    ligne_pred.set_data(x, U[m, nodes])
    ligne_err.set_data(x, np.abs(U[m, nodes] - U_reel[m, nodes]))
    titre.set_text(f"Propagation de l'onde — t = {m*dt:.3f}  (pas {m})")
    return ligne_reel, ligne_pred, ligne_err, titre

anim = animation.FuncAnimation(fig_anim, maj, frames=frames, interval=50, blit=False)
anim.save(OUTPUT_DIR / "propagation_onde.gif", writer="pillow", fps=20, dpi=110)
plt.close(fig_anim)
print(f"Animation sauvegardée : {OUTPUT_DIR / 'propagation_onde.gif'}")

# (b) Erreur en fonction du temps, sur les indices reellement remplis
steps   = np.arange(2*ndt, Nt + 1, ndt)
t_axis  = steps * dt
l2_list   = [l2_rel(U[k, nodes], U_reel[k, nodes])        for k in steps]
linf_list = [np.max(np.abs(U[k, nodes] - U_reel[k, nodes])) for k in steps]

plt.figure(figsize=(9, 5))
plt.plot(t_axis, l2_list,   "o-", ms=3, label="erreur L2 relative")
plt.plot(t_axis, linf_list, "s-", ms=3, label="erreur max absolue (Linf)")
plt.yscale("log")
plt.xlabel("t"); plt.ylabel("erreur"); plt.grid(True, which="both"); plt.legend()
plt.title("Erreur du rollout en fonction du temps")
plt.savefig(OUTPUT_DIR / "erreur_temps.png", dpi=150, bbox_inches="tight")
plt.close()

import openpyxl
xlsx_path = "/Users/aloishenom/Research_Project/Code/Comparaisons/comparative_table.xlsx"
erreur = l2_list      
wb = openpyxl.load_workbook(xlsx_path)
ws = wb.active          
col_uxx = None
for cell in ws[1]:
    if cell.value == "Ut_Uxx":
        col_uxx = cell.column
        break
if col_uxx is None:
    raise ValueError("Colonne 'Ut_Uxx' introuvable dans la première ligne.")
col_time = col_uxx - 1 
for i, (t, e) in enumerate(zip(t_axis, erreur), start=2):
    ws.cell(row=i, column=col_time, value=float(t))
    ws.cell(row=i, column=col_uxx,  value=float(e))
wb.save(xlsx_path)
print(f"{len(erreur)} valeurs écrites dans {xlsx_path} (colonnes time / Ut_Uxx).")


smape_list = [100.0 * smape(U[k, nodes], U_reel[k, nodes]) for k in steps]  

plt.figure(figsize=(9, 5))
plt.plot(t_axis, smape_list, "s-", ms=3, label="sMAPE")
plt.xlabel("t"); plt.ylabel("erreur (%)"); plt.grid(True); plt.legend()
plt.title("sMAPE du rollout en fonction du temps")
plt.savefig(OUTPUT_DIR / "smape_temps.png", dpi=150, bbox_inches="tight")
plt.close()