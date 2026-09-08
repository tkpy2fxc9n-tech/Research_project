from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

import matplotlib
matplotlib.use("Agg")  # backend non-interactif, pas de fenêtre qui s'ouvre

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
i_left, i_right = SS, Ntot - SS

t_end=3
dt=t_end/Nt
dx = L / (Nx - 1)
CFL = dt/dx * np.sqrt(E/rho)

#Parameters for the roll out
ndt = 5
M_BACK = 3     # niveaux temporels en entrée : t, t-ndt, ..., t-(M_BACK-1)*ndt
N_FWD  = 3     # horizons de sortie : n+ndt, n+2ndt, ..., n+N_FWD*ndt
SMOOTH_ALPHA = 0.20     # lissage Laplacien au rollout ; doit rester < 0.25

N=6
LAG_LABELS = {0: "t", 1: "t-dt", 2: "t-2dt"}
def jlabel(k):
    return "j" if k == 0 else f"j{k:+d}"

AMPLITUDES = np.linspace(0.005, 0.1, N).round(3).tolist()
PULSATIONS = np.linspace(1, 10, N).round(1).tolist()
n_sims = len(AMPLITUDES)*len(PULSATIONS)

# We compute each simulation
all_dfs = []
FIELDS = {}          # champ complet u(t,x) de chaque simulation, pour le pushforward

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

    FIELDS[(A, omega)] = u_storage          # champ complet pour le pushforward

    # --- Construction du dataset : entree = U brut sur 3 instants (t, t-ndt, t-2ndt), voisinage ±SS ---
    rows = []

    for n in range(ndt, Nt - ndt + 1):
        row = {"A": A, "omega": omega, "n_step": n}
        u_t  = u_storage[n]           # présent
        u_tm = u_storage[n - ndt]     # passé (t - ndt)

        for k in range(-SS, SS + 1):                 # 11 voisins à t  → u_xx
            row[f"u_t({jlabel(k)})"] = u_t[nodes + k]
        row["u_tm(j)"] = u_tm[nodes]                 # centre à t-ndt  → (avec u_t) u_dot

        row["delta_u"] = u_storage[n + ndt, nodes] - u_t[nodes]   # cible
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

OUTPUTS = ["delta_u"]      
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
N_EPOCHS      = 15
BATCH_SIZE    = 512

# Pushforward (stabilisation du rollout)
NOISE_STD    = 0.10   # bruit ajouté aux entrées (ta valeur d'origine)
LAMBDA_PF    = 1.0    # poids de la loss pushforward
N_PF_GROUPS  = 8      # nombre de simulations déroulées par batch
PF_WARMUP    = 2      # montée progressive de LAMBDA_PF sur les premières époques

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

# ============================================================
# OUTILS POUR LE PUSHFORWARD (Brandstetter et al. 2022)
# On déroule le réseau d'UN bloc sur sa propre prédiction (gradient détaché),
# puis on lui demande de revenir vers la vérité au bloc suivant.
# ============================================================
mu_in  = norm_stats.loc[INPUTS,  "mean"].values.astype(np.float32)
sd_in  = norm_stats.loc[INPUTS,  "std" ].values.astype(np.float32)
mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
sd_out = norm_stats.loc[OUTPUTS, "std" ].values.astype(np.float32)

def u_right_val(A, omega, t):
    return A * np.sin(omega * t) if t < 2 * np.pi / omega else 0.0

def build_window(u_t, u_tm):
    X = np.zeros((len(nodes), len(INPUTS)), np.float32)
    col = 0
    for off in range(-SS, SS + 1):
        X[:, col] = u_t[nodes + off]; col += 1
    X[:, col] = u_tm[nodes]; col += 1
    return (X - mu_in) / sd_in

def reconstruct(u_curr, n_curr, pred_norm, A, omega):
    """Depuis le delta prédit (normalisé), reconstruit le champ prédit à t+ndt."""
    delta = pred_norm[:, 0] * sd_out[0] + mu_out[0]
    s = n_curr + ndt
    uu = np.zeros(Ntot)
    uu[nodes]     = u_curr[nodes] + delta
    uu[:i_left+1] = 0.0
    uu[i_right:]  = u_right_val(A, omega, s * dt)
    return uu

# il faut pouvoir superviser jusqu'à n + 2*ndt
PF_SAMPLES = [(A, omega, n)
              for (A, omega) in FIELDS
              for n in range(ndt, Nt - 2*ndt + 1)]
print(f"Échantillons pushforward disponibles : {len(PF_SAMPLES):,}")

def pushforward_loss(n_groups):
    idxs = np.random.choice(len(PF_SAMPLES), n_groups, replace=False)
    groups = [PF_SAMPLES[i] for i in idxs]

    # Bloc 1 : entrées vraies, détachées -> champ abîmé à n+ndt
    X1 = np.concatenate(
        [build_window(FIELDS[(A, omega)][n], FIELDS[(A, omega)][n - ndt])
         for (A, omega, n) in groups], axis=0)
    with torch.no_grad():
        pred1 = modele(torch.tensor(X1)).numpy()

    nN = len(nodes)
    X2_list, tgt_list = [], []
    for j, (A, omega, n) in enumerate(groups):
        U  = FIELDS[(A, omega)]
        Up = reconstruct(U[n], n, pred1[j*nN:(j+1)*nN], A, omega)   # abîmé à n+ndt

        nprime = n + ndt                                # nouveau présent
        X2_list.append(build_window(Up, U[n]))          # voisins abîmés + centre passé vrai

        curr = Up[nodes]
        tgt  = (U[nprime + ndt][nodes] - curr)[:, None] # cible = vérité(n+2ndt) - abîmé
        tgt_list.append(((tgt - mu_out) / sd_out).astype(np.float32))

    pred2 = modele(torch.tensor(np.concatenate(X2_list, axis=0)))
    return criterion(pred2, torch.tensor(np.concatenate(tgt_list, axis=0)))


historique_train = []
historique_val   = []
historique_pf    = []

meilleure_val = float("inf")   # on initialise au pire score possible

for epoch in range(1, N_EPOCHS + 1):

    # Montée progressive du poids pushforward (le modèle est nul au début)
    lam_pf = LAMBDA_PF * min(1.0, epoch / PF_WARMUP)

    # Phase entraînement
    modele.train()
    perte_train    = 0.0       #initialisation of training loss
    perte_pf_total = 0.0
    for X_batch, y_batch in train_loader:
        optimiseur.zero_grad()                       # remet les gradients à zéro

        # --- Loss de données : entrée bruitée, cible propre ---
        X_in       = X_batch + NOISE_STD * torch.randn_like(X_batch)
        prediction = modele(X_in)                    # prédiction
        data_loss  = criterion(prediction, y_batch)  # erreur

        # --- Loss pushforward (stabilise le rollout) ---
        pf_loss = pushforward_loss(N_PF_GROUPS) if lam_pf > 0 else torch.tensor(0.0)

        total_loss = data_loss + lam_pf * pf_loss
        total_loss.backward()                        # rétropropagation
        optimiseur.step()                            # mise à jour des poids

        perte_train    += data_loss.item()
        perte_pf_total += pf_loss.item()
    perte_train    /= len(train_loader)
    perte_pf_total /= len(train_loader)

    # Phase validation : un seul forward pass sur tout le val
    modele.eval()
    with torch.no_grad():
        pred_val = modele(torch.tensor(X_val)).numpy()

    # Loss globale 
    perte_val = ((pred_val - y_val)**2).mean()
    scheduler.step(perte_val)

    historique_train.append(perte_train)
    historique_val  .append(perte_val)
    historique_pf   .append(perte_pf_total)

    if epoch % 2 == 0:
        print(f"Epoch {epoch:4d}/{N_EPOCHS}  —  "
            f"data: {perte_train:.4f}  |  pushf: {perte_pf_total:.4f}  |  val: {perte_val:.4f}")
    
    # Si la val s'est améliorée, on sauvegarde ce modèle
    if perte_val < meilleure_val:
        meilleure_val = perte_val
        torch.save(modele.state_dict(), SCRIPT_DIR / "model.pth")   # on écrase avec le meilleur

# Courbe d'apprentissage
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(historique_train, label="Data (train)")
ax.plot(historique_val,   label="Data (val)")
ax.plot(historique_pf,    label="Pushforward")
ax.set_xlabel("Epoch") ; ax.set_ylabel("Loss")
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
mu_in = norm_stats.loc[INPUTS, "mean"].values.astype(np.float32)   # une valeur par feature U
sd_in = norm_stats.loc[INPUTS, "std"].values.astype(np.float32)    # une valeur par feature U

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

history_needed = ndt
print(history_needed, 'history needed')

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

for n in range(history_needed, Nt - ndt + 1, ndt):
    X = build_window(U[n], U[n - ndt])
    with torch.no_grad():
        sortie = modele(torch.tensor(X)).numpy()
    delta = sortie[:, 0] * sd_out[0] + mu_out[0] - biais_repos
    s = n + ndt
    U[s, nodes]     = U[n, nodes] + delta
    U[s, :i_left+1] = 0.0
    U[s, i_right:]  = u_right(s * dt)

    if SMOOTH_ALPHA > 0:
        j0, j1 = i_left + 1, i_right           
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

from matplotlib.animation import FuncAnimation, PillowWriter

frames_scatter = list(range(ndt, Nt - ndt + 1, ndt))

fig, ax = plt.subplots(figsize=(6, 6))
nuage = ax.scatter([], [], s=10)
ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
ax.set_xlabel("u_xx (predit)"); ax.set_ylabel("u_tt (predit)")
ax.grid(True)
titre = ax.set_title("")

def init():
    nuage.set_offsets(np.empty((0, 2)))
    return (nuage,)

def update(n):
    xx_vals = upred_xx_dict[n, i_left+1:i_right]
    tt_vals = upred_tt_dict[n, i_left+1:i_right]
    nuage.set_offsets(np.column_stack([xx_vals, tt_vals]))
    titre.set_text(f"t = {n * dt:.2f}")
    return (nuage,)

ani_scatter = FuncAnimation(fig, update, frames=frames_scatter, init_func=init,
                            blit=False, interval=100)

ani_scatter.save(OUTPUT_DIR / "scatter_utt_uxx_predit.gif", writer=PillowWriter(fps=10))
plt.close(fig)

print("temps physique :", round(time_phys, 6))
print("temps predit   :", round(time_pred, 6))

m =  70 # multiple de ndt, dans la zone ou tu vois l'ecart
print("bord droit reel  :", U_reel[m, i_right:])
print("bord droit predit:", U[m, i_right-4:i_right])
print("interieur reel   :", U_reel[m, i_right-4:i_right])
print("interieur predit :", U[m + history_needed, i_right-4:i_right])

# =====================================================
# 3) Erreurs
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

# (a) Champ reel/predit + erreur spatiale a un instant cible
t_cible = 0.3
m = int(round(t_cible / dt))
m = (m // ndt) * ndt
print(f"trace a t={m*dt:.3f} (indice {m})")

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

smape_list = [100.0 * smape(U[k, nodes], U_reel[k, nodes]) for k in steps]  

plt.figure(figsize=(9, 5))
plt.plot(t_axis, smape_list, "s-", ms=3, label="sMAPE")
plt.xlabel("t"); plt.ylabel("erreur (%)"); plt.grid(True); plt.legend()
plt.title("sMAPE du rollout en fonction du temps")
plt.savefig(OUTPUT_DIR / "smape_temps.png", dpi=150, bbox_inches="tight")
plt.close()

# Graph animated

# 1) images : uniquement les indices remplis (multiples de ndt)
frames = list(range(0, Nt + 1, ndt))

# 2) figure a deux panneaux + courbes vides
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
ligne_reel,   = ax1.plot([], [], "r",  lw=2, label="reel")
ligne_predit, = ax1.plot([], [], "b--", lw=2, label="predit")
ligne_erreur, = ax2.plot([], [], "k",  lw=1.5, label="|predit - reel|")

# 3) axes fixes
ax1.set_xlim(0, L)
ymax = np.abs(U_reel[:, nodes]).max() * 1.1
ax1.set_ylim(-ymax, ymax)
ax1.set_ylabel("u"); ax1.legend(loc="upper left"); ax1.grid(True)

frames_arr = np.arange(0, Nt + 1, ndt)
err_all = np.abs(U[frames_arr][:, nodes] - U_reel[frames_arr][:, nodes])
emax = err_all.max() * 1.1
ax2.set_xlim(0, L)
ax2.set_ylim(0, emax)
ax2.set_xlabel("x"); ax2.set_ylabel("erreur absolue"); ax2.legend(loc="upper left"); ax2.grid(True)

titre = fig.suptitle("")

# 4) init + update
def init():
    ligne_reel.set_data([], [])
    ligne_predit.set_data([], [])
    ligne_erreur.set_data([], [])
    return ligne_reel, ligne_predit, ligne_erreur

def update(frame):
    ligne_reel.set_data(x, U_reel[frame, nodes])
    ligne_predit.set_data(x, U[frame, nodes])
    ligne_erreur.set_data(x, np.abs(U[frame, nodes] - U_reel[frame, nodes]))
    titre.set_text(f"t = {frame * dt:.2f}")
    return ligne_reel, ligne_predit, ligne_erreur

# 5) animation
ani = FuncAnimation(fig, update, frames=frames, init_func=init,
                    blit=False, interval=100)

ani.save(OUTPUT_DIR / "rollout_animation.gif", writer=PillowWriter(fps=10))
plt.close(fig)