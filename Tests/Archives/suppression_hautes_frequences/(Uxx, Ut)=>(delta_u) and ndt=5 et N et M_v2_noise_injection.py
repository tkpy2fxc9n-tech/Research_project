# ============================================================
# VARIANTE : v2_noise_injection — bruit ciblé sur le mode haute fréquence
#
# Méthode testée (ISOLÉE, tout le reste = baseline) :
#   Le script original injecte DÉJÀ du bruit gaussien à l'entraînement
#   (NOISE_STD=0.10, ligne ~350 de la baseline) : X_in = X_batch +
#   NOISE_STD * randn_like(X_batch), c'est-à-dire un bruit BLANC isotrope
#   (i.i.d.) sur les 132 features normalisées (u_dot et u_xx, aux 3 lags
#   temporels, sur les 23 voisins -SS..+SS). Un bruit blanc a un spectre
#   plat : il contient de la haute fréquence, mais mélangée à de la basse
#   fréquence, donc il n'entraîne pas spécifiquement le réseau à rejeter le
#   mode damier (Nyquist spatial) qui pose problème.
#
#   Ici, on REMPLACE ce bruit blanc par un bruit STRUCTURÉ, concentré
#   exactement sur le mode de Nyquist spatial (le damier lui-même) :
#     pattern(k) = (-1)^k  sur l'axe des voisins k=-SS..+SS  (alterné)
#     bruit(échantillon, feature) = eps_échantillon * pattern(feature)
#     avec eps_échantillon ~ N(0, NOISE_STD^2), une amplitude aléatoire
#     tirée UNE FOIS par échantillon (pas par feature), pour bruiter chaque
#     fenêtre d'entrée par une réplique aléatoire du damier plutôt que par
#     un bruit indépendant par feature.
#   Le pattern est dupliqué (u_dot, u_xx) puis répété sur les M_BACK lags,
#   dans le même ordre de colonnes que build_window() (vérifié contre la
#   génération du dataset : pour chaque lag, pour chaque voisin k, on a
#   d'abord u_dot(k) puis u_xx(k)).
#   NOISE_STD garde la même valeur (0.10) que la baseline pour une
#   comparaison à amplitude de bruit égale — seule la FORME (spectre) du
#   bruit change.
# SMOOTH_ALPHA (0.20) et LAMBDA_PF (1.0) restent aux valeurs de la baseline.
#
# Seuls changements par rapport à l'original (identiques dans les 6 copies,
# nécessaires pour comparer les méthodes sans qu'elles s'écrasent entre
# elles et sans dépendre d'un fichier externe cassé) :
#   - dossier de sortie et modèle sauvegardés séparément par variante
#     (outputs_<variant>/) au lieu du dossier "outputs" partagé,
#   - ajout du RMSE par pas (en plus de L2 relative / Linf / sMAPE déjà
#     présents), d'un seuil de divergence et d'un spectre FFT de l'erreur,
#   - export d'un metrics.json par variante, lu par Comparatifs/compare_methodes.py,
#   - suppression des blocs qui ne servent pas à CETTE comparaison (vérif EDP
#     u_tt/u_xx, animation GIF, éval teacher-forcing sur le jeu de test,
#     benchmark de vitesse FD vs NN) — ils faisaient perdre du temps de calcul
#     pour des sorties non utilisées par Comparatifs/compare_methodes.py,
#   - suppression du bloc qui écrivait dans
#     Code_comparaison_des_inputs/Comparaisons/comparative_table.xlsx : ce
#     fichier appartient à une AUTRE étude d'ablation (comparaison des champs
#     d'entrée U/Ut/Uxx) et le chemin relatif utilisé ici
#     (SCRIPT_DIR.parent / "Comparaisons") ne correspond même plus à
#     l'emplacement actuel du script — il aurait fait planter le run avant
#     resume.txt. Voir RESUME_CHANGEMENTS.md à la racine du dossier.
# ============================================================

# ============================================================
# IMPORTS (tout regroupé ici)
# ============================================================
from pathlib import Path
import time
import json
from itertools import product

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # backend non-interactif : aucune fenêtre ne s'ouvre
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

VARIANT_NAME = "v2_noise_injection"

# Dossier où sont sauvegardés tous les graphes (un dossier par variante pour
# ne pas écraser les sorties des autres copies de l'ablation)
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / f"outputs_{VARIANT_NAME}"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# PARAMÈTRES
# ============================================================
# Physique
E   = 1
rho = 2
L   = 1

# Grille espace / temps
Nt   = 500
Nx   = 100
SS   = 11                       
Ntot = Nx + 2*SS                  

nodes = np.arange(SS, Ntot-SS)   

t_end = 5
dt = t_end / Nt
dx = L / (Nx - 1)
CFL = dt/dx * np.sqrt(E/rho)

# Rollout : M instants passés -> N instants futurs (espacés de ndt pas)
ndt    = 3
M_BACK = 3     # niveaux temporels en entrée  : t, t-ndt, ..., t-(M_BACK-1)*ndt
N_FWD  = 3     # horizons de sortie           : n+ndt, n+2ndt, ..., n+N_FWD*ndt

# --- Comparaison inter-méthodes (identique dans les 6 copies de l'ablation) ---
# Seuil de divergence : premier pas de rollout où le RMSE dépasse cette
# fraction de l'amplitude d'entrée A du run (RMSE a les mêmes unités physiques
# que u, donc une fraction de A est un seuil interprétable et transférable
# d'un (A, omega) à l'autre). Aucun seuil de ce type n'existait déjà ailleurs
# dans le projet (cherché dans Code_comparaison_des_inputs, Code, Old Code) :
# valeur choisie ici, à ajuster si besoin — la courbe RMSE(t) complète est de
# toute façon sauvegardée, donc le seuil peut être recalculé a posteriori.
DIVERGENCE_THRESHOLD_FRAC = 0.5
# Pas de rollout (indice temporel n, multiple de ndt) utilisé pour le spectre
# FFT comparatif de l'erreur entre les 6 variantes. ~60% de l'horizon de
# rollout : assez tard pour laisser le mode damier se développer, mais fixe
# et identique pour les 6 runs (A est déterministe et identique partout).
COMPARISON_STEP = 300

def jlabel(k):                    # libellé de colonne pour le voisin j+k
    return "j" if k == 0 else f"j{k:+d}"

# Jeu de simulations : N valeurs d'amplitude × N valeurs de pulsation
N = 6                    
AMPLITUDES = np.linspace(0.005, 0.1, N).round(3).tolist()
PULSATIONS = np.linspace(3, 10, N).round(1).tolist()


# ============================================================
# GÉNÉRATION DES DONNÉES
# Pour chaque (A, omega) : on simule l'onde par différences finies,
# puis on en extrait les entrées (M_BACK niveaux) et sorties (N_FWD horizons).
# ============================================================
all_dfs = []
FIELDS = {}          # champ complet u(t,x) de chaque simulation, pour le pushforward

def u_right_val(A, omega, t):
    sigma = np.interp(omega, [1.0, 10.0], [0.15, 0.07])   
    t0    = 4.0 * sigma                                  
    return A * np.exp(-((t - t0) / sigma) ** 2)

for A, omega in product(AMPLITUDES, PULSATIONS):  

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
        u_new[i_right:] = u_right_val(A, omega, t + dt)

        u_xx_storage[n] = u_xx
        u_1 = u.copy()
        u   = u_new
        u_storage[n + 1] = u.copy()

    FIELDS[(A, omega)] = u_storage          # on garde le champ complet pour le pushforward

    # --- Construction du dataset : M_BACK instants passés, voisinage ±SS ---
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


# ============================================================
# SPLIT train / val / test  +  NORMALISATION
# ============================================================
rng = np.random.default_rng(seed=42)

n_rows  = len(df)
n_train = int(0.90 * n_rows)
n_val   = int(0.05 * n_rows)
n_test  = n_rows - n_train - n_val

split_labels = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
rng.shuffle(split_labels)
df["split"] = split_labels

print("Distribution du split :")
for s in ["train", "val", "test"]:
    n = (df["split"] == s).sum()
    print(f"  {s:5s} : {n:>8,} lignes  ({100*n/len(df):.1f} %)")

# Colonnes d'entrée / sortie
OUTPUTS = [f"delta_u@{h}ndt" for h in range(1, N_FWD + 1)]
meta    = ["A", "omega", "n_step", "split"]
INPUTS  = [c for c in df.columns if c not in meta + OUTPUTS]

# Stats de normalisation calculées sur le TRAIN uniquement
train_mask = df["split"] == "train"
norm_stats = pd.DataFrame({
    "mean": df.loc[train_mask, INPUTS + OUTPUTS].mean(),
    "std" : df.loc[train_mask, INPUTS + OUTPUTS].std(),
})
norm_stats["std"] = norm_stats["std"].replace(0, 1)

# Application de la normalisation à tout le dataset (colonnes suffixées "_n")
for col in INPUTS + OUTPUTS:
    df[col + "_n"] = (df[col] - norm_stats.loc[col, "mean"]) / norm_stats.loc[col, "std"]


# ============================================================
# HYPERPARAMÈTRES
# ============================================================
# Architecture du réseau
HIDDEN_SIZES = [64, 32, 16]

# Entraînement
LEARNING_RATE = 1e-3
N_EPOCHS      = 20
BATCH_SIZE    = 512

# Pushforward (stabilisation du rollout)
LAMBDA_PF   = 1.0   # poids de la loss pushforward (même échelle que la loss données)
N_PF_GROUPS = 8     # nombre de simulations (champs complets) déroulées par batch
PF_WARMUP   = 2     # montée progressive de LAMBDA_PF sur les premières époques

# Robustification du rollout (distribution shift). Le rollout consomme ses propres
# prédictions ; u_xx (dérivée 2nde, ÷dx² ≈ ×1e4) amplifie le mode "damier" haute
# fréquence -> divergence. Deux parades :
#   1) NOISE_STD : on bruite les features (normalisées) à l'entraînement, donc le
#      réseau apprend à débruiter ses propres entrées.
#   2) SMOOTH_ALPHA : un pas de lissage Laplacien sur le champ prédit à chaque bloc
#      du rollout, qui amortit le damier (×(1-4α)) sans toucher l'onde (basse fréq).
NOISE_STD    = 0.10
SMOOTH_ALPHA = 0.20

print(f"Dataset : {len(df):,} lignes")
print(f"Splits  : {df['split'].value_counts().to_dict()}")


# ============================================================
# TENSEURS PyTorch + DataLoader d'entraînement
# ============================================================
INPUTS_N  = [f"{x}_n" for x in INPUTS]
OUTPUTS_N = [f"{x}_n" for x in OUTPUTS]

X_train = df.loc[df["split"] == "train", INPUTS_N].values.astype(np.float32)
y_train = df.loc[df["split"] == "train", OUTPUTS_N].values.astype(np.float32)

X_val   = df.loc[df["split"] == "val", INPUTS_N].values.astype(np.float32)
y_val   = df.loc[df["split"] == "val", OUTPUTS_N].values.astype(np.float32)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                          batch_size=BATCH_SIZE, shuffle=True)

# --- v2 : masque "damier" (mode de Nyquist spatial) dans l'espace des
# features d'entrée normalisées, pour bruiter spécifiquement ce mode plutôt
# qu'un bruit blanc isotrope (voir explication en tête de fichier). Ordre
# des colonnes = ordre exact de build_window()/génération du dataset :
# pour chaque lag, pour chaque voisin k=-SS..+SS : u_dot(k) puis u_xx(k).
_offset_sign  = np.array([(-1.0) ** i for i in range(2*SS + 1)], dtype=np.float32)
_checker_bloc = np.repeat(_offset_sign, 2)          # dupliqué pour (u_dot, u_xx)
checker_mask  = np.tile(_checker_bloc, M_BACK)      # répété sur les M_BACK lags
assert len(checker_mask) == len(INPUTS), "checker_mask ne correspond pas à l'ordre des colonnes de INPUTS"
checker_mask_t = torch.tensor(checker_mask)


# ============================================================
# OUTILS POUR LE PUSHFORWARD
#
# Le rollout diverge à cause du *distribution shift* : à l'entraînement le réseau
# voit des entrées propres (teacher forcing), mais au rollout il consomme ses
# propres prédictions, ré-injectées via u_xx (amplificateur de bruit). Le
# pushforward (Brandstetter et al. 2022) corrige ça : on déroule le réseau d'UN
# bloc sur sa propre prédiction (gradient détaché), puis on lui demande de revenir
# vers la vérité au bloc suivant.
# ============================================================
i_left, i_right = SS, Ntot - SS

# Moyennes / écarts-types, dans l'ordre EXACT de INPUTS / OUTPUTS
mu_in  = norm_stats.loc[INPUTS,  "mean"].values.astype(np.float32)
sd_in  = norm_stats.loc[INPUTS,  "std" ].values.astype(np.float32)
mu_out = norm_stats.loc[OUTPUTS, "mean"].values.astype(np.float32)
sd_out = norm_stats.loc[OUTPUTS, "std" ].values.astype(np.float32)



def u_xx_field(u):
    """u_xx sur la grille étendue (zéro hors [i_left, i_right]), comme le dataset."""
    out = np.zeros(Ntot)
    out[i_left:i_right+1] = (u[i_left-1:i_right] - 2*u[i_left:i_right+1] + u[i_left+1:i_right+2]) / dx**2
    return out

def build_window(m_list, field_at):
    X = np.zeros((len(nodes), len(INPUTS)), np.float32)
    col = 0
    for m in m_list:
        udot = (field_at(m) - field_at(m - ndt)) / (ndt * dt)
        uxx  = u_xx_field(field_at(m))
        for off in range(-SS, SS + 1):
            X[:, col] = udot[nodes + off]; col += 1
            X[:, col] = uxx[nodes + off];  col += 1
    return (X - mu_in) / sd_in

def reconstruct(u_curr, n_curr, pred_norm, A, omega):
    deltas = pred_norm * sd_out + mu_out            
    champs = {}
    for h in range(1, N_FWD + 1):
        s = n_curr + h * ndt
        u = np.zeros(Ntot)
        u[nodes]     = u_curr[nodes] + deltas[:, h-1]
        u[:i_left+1] = 0.0
        u[i_right:]  = u_right_val(A, omega, s * dt)
        champs[s] = u
    return champs


PF_SAMPLES = [(A, omega, n)
              for (A, omega) in FIELDS
              for n in range(M_BACK*ndt, Nt - 2*N_FWD*ndt + 1)]
print(f"Échantillons pushforward disponibles : {len(PF_SAMPLES):,}")


# ============================================================
# MODÈLE
# ============================================================
class Reseau(nn.Module):

    def __init__(self, n_inputs, n_outputs, hidden_sizes):
        super().__init__()
        couches = []
        taille_entree = n_inputs
        for taille in hidden_sizes:
            couches.append(nn.Linear(taille_entree, taille))
            couches.append(nn.GELU())
            taille_entree = taille
        couches.append(nn.Linear(taille_entree, n_outputs))
        self.reseau = nn.Sequential(*couches)

    def forward(self, x):
        return self.reseau(x)

modele = Reseau(n_inputs=len(INPUTS_N), n_outputs=len(OUTPUTS_N), hidden_sizes=HIDDEN_SIZES)
print(modele)
n_params = sum(p.numel() for p in modele.parameters())
print(f"\nNombre de paramètres : {n_params:,}")

criterion  = nn.MSELoss()
optimiseur = torch.optim.Adam(modele.parameters(), lr=LEARNING_RATE)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiseur, mode="min", factor=0.5, patience=10)

def pushforward_loss(n_groups):
    idxs = np.random.choice(len(PF_SAMPLES), n_groups, replace=False)
    groups = [PF_SAMPLES[i] for i in idxs]

    X1 = np.concatenate([build_window([n, n-ndt, n-2*ndt], lambda m, U=FIELDS[(A, omega)]: U[m]) for (A, omega, n) in groups], axis=0,)
    with torch.no_grad():
        pred1 = modele(torch.tensor(X1)).numpy()

    nN = len(nodes)
    X2_list, tgt_list = [], []
    for j, (A, omega, n) in enumerate(groups):
        U  = FIELDS[(A, omega)]
        Up = reconstruct(U[n], n, pred1[j*nN:(j+1)*nN], A, omega)   
        field_at = lambda m, U=U, Up=Up: Up[m] if m in Up else U[m]

        nprime = n + N_FWD * ndt                                   
        X2_list.append(build_window([nprime, nprime-ndt, nprime-2*ndt], field_at))

        curr = Up[nprime][nodes]                                   
        tgt  = np.stack([U[nprime + h*ndt][nodes] - curr for h in range(1, N_FWD+1)], axis=1)
        tgt_list.append(((tgt - mu_out) / sd_out).astype(np.float32))

    pred2 = modele(torch.tensor(np.concatenate(X2_list, axis=0)))  
    return criterion(pred2, torch.tensor(np.concatenate(tgt_list, axis=0)))

# ============================================================
# ENTRAÎNEMENT  (loss données + bruit + pushforward)
# ============================================================
historique_train = []
historique_val   = []
historique_pf    = []
meilleure_val    = float("inf")

t_train_start = time.perf_counter()

for epoch in range(1, N_EPOCHS + 1):

    # Montée progressive du poids pushforward (le modèle est nul au début)
    lam_pf = LAMBDA_PF * min(1.0, epoch / PF_WARMUP)

    modele.train()
    perte_train    = 0.0
    perte_pf_total = 0.0

    for X_batch, y_batch in train_loader:
        optimiseur.zero_grad()

        # --- Loss de données : on bruite l'entrée mais la cible reste le delta
        #     PROPRE -> le réseau apprend à bien prédire même sur entrées abîmées.
        # v2 : bruit structuré concentré sur le mode damier (Nyquist spatial)
        #     plutôt que bruit blanc isotrope — voir explication en tête de fichier. ---

        if NOISE_STD > 0:
            eps  = NOISE_STD * torch.randn(X_batch.shape[0], 1)   # amplitude aléatoire par échantillon
            X_in = X_batch + eps * checker_mask_t                  # bruit concentré au mode de Nyquist spatial
        else:
            X_in = X_batch

        prediction = modele(X_in)
        data_loss  = criterion(prediction, y_batch)

        # --- Loss pushforward (stabilise le rollout) ---
        if lam_pf > 0:
            pf_loss = pushforward_loss(N_PF_GROUPS)
        else:
            pf_loss = torch.tensor(0.0)

        total_loss = data_loss + lam_pf * pf_loss
        total_loss.backward()
        optimiseur.step()

        perte_train    += data_loss.item()
        perte_pf_total += pf_loss.item()

    perte_train    /= len(train_loader)
    perte_pf_total /= len(train_loader)

    modele.eval()
    with torch.no_grad():
        pred_val = modele(torch.tensor(X_val)).numpy()
    perte_val = ((pred_val - y_val)**2).mean()
    scheduler.step(perte_val)

    historique_train.append(perte_train)
    historique_val  .append(perte_val)
    historique_pf   .append(perte_pf_total)

    if epoch % 1 == 0:
        print(f"Epoch {epoch:4d}/{N_EPOCHS}  —  "
              f"data: {perte_train:.4f}  |  pushf: {perte_pf_total:.4f}  |  val: {perte_val:.4f}")

    # On garde le meilleur modèle (val minimale)
    if perte_val < meilleure_val:
        meilleure_val = perte_val
        torch.save(modele.state_dict(), OUTPUT_DIR / "model.pth")

train_time_s = time.perf_counter() - t_train_start

# --- GRAPHE : courbe d'apprentissage ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(historique_train, label="Data (train)")
ax.plot(historique_val,   label="Data (val)")
ax.plot(historique_pf,    label="Pushforward")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Courbe d'apprentissage (pushforward)")
ax.set_yscale("log"); ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "courbe_apprentissage.png", dpi=150, bbox_inches="tight")
plt.close()

# On recharge le meilleur modèle trouvé pendant l'entraînement
modele.load_state_dict(torch.load(OUTPUT_DIR / "model.pth", weights_only=True))
print(f"Meilleur modèle rechargé — val minimale : {meilleure_val:.6f}")


# ============================================================
# ROLLOUT — paramètres choisis pour rejouer une propagation
# ============================================================
A, omega = AMPLITUDES[0], PULSATIONS[0]
x = np.linspace(0, L, Nx)

def u_right(t):
    return u_right_val(A, omega, t)

def u_xx_etendu(u):                  # u_xx sur la grille étendue (longueur Ntot)
    u_xx = np.zeros(Ntot)
    u_xx[i_left:i_right+1] = (u[i_left-1:i_right] - 2*u[i_left:i_right+1] + u[i_left+1:i_right+2]) / dx**2
    return u_xx

# Biais "au repos" : sortie du réseau pour une entrée nulle. On le soustrait au
# rollout pour que la zone au repos reste bien à zéro.
mu = norm_stats.loc[OUTPUTS, "mean"].values
sd = norm_stats.loc[OUTPUTS, "std"].values
Xz = (np.zeros((len(nodes), len(INPUTS)), dtype=np.float32) - mu_in) / sd_in
with torch.no_grad():
    biais_repos = (modele(torch.tensor(Xz)).numpy() * sd + mu)[0]

# ----------------------------------------------------------------
# 1) VRAIE simulation (référence)
# ----------------------------------------------------------------
t0 = time.perf_counter()

U_reel = np.zeros((Nt + 1, Ntot))
u, u_1 = np.zeros(Ntot), np.zeros(Ntot)

for n in range(Nt):
    u_new = np.zeros(Ntot)
    u_new[i_left:i_right+1] = (2*u[i_left:i_right+1] - u_1[i_left:i_right+1] + CFL**2 * (u[i_left-1:i_right] - 2*u[i_left:i_right+1] + u[i_left+1:i_right+2]))
    u_new[:i_left+1] = 0.0
    u_new[i_right:]  = u_right((n+1)*dt)
    u_1, u = u.copy(), u_new
    U_reel[n + 1] = u.copy()

time_phys = time.perf_counter() - t0

# ----------------------------------------------------------------
# 2) Simulation PRÉDITE (rollout du réseau sur ses propres sorties)
# ----------------------------------------------------------------
t0 = time.perf_counter()

history_needed = M_BACK * ndt
U = np.zeros((Nt + 1, Ntot))
for m in range(history_needed + 1):      # on amorce avec la vérité
    U[m] = U_reel[m]

for n in range(history_needed, Nt - N_FWD * ndt + 1, N_FWD*ndt):

    udot = [(U[n - lag*ndt] - U[n - (lag+1)*ndt]) / (ndt*dt) for lag in range(M_BACK)]
    uxx  = [u_xx_etendu(U[n - lag*ndt])                      for lag in range(M_BACK)]

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

    # On écrit les N_FWD horizons prédits, conditions aux limites + lissage
    for h in range(1, N_FWD + 1):
        s = n + h*ndt
        U[s, nodes]     = U[n, nodes] + deltas[:, h-1]
        U[s, :i_left+1] = 0.0
        U[s, i_right:] = u_right_val(A, omega, s * dt)

        # Lissage Laplacien léger : casse le mode "damier" haute fréquence que
        # u_xx amplifierait au bloc suivant, sans amortir l'onde (basse fréq).
        if SMOOTH_ALPHA > 0:
            j0, j1 = i_left + 1, i_right            # nœuds intérieurs (hors bords)
            lap = U[s, j0-1:j1-1] - 2*U[s, j0:j1] + U[s, j0+1:j1+1]
            U[s, j0:j1] += SMOOTH_ALPHA * lap

time_pred = time.perf_counter() - t0

print("temps physique :", round(time_phys, 6))
print("temps predit   :", round(time_pred, 6))

# ============================================================
# GRAPHES FINAUX DU ROLLOUT
# ============================================================
def l2_rel(pred, true, eps=1e-12):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps)

def rmse(pred, true):
    return np.sqrt(np.mean((pred - true) ** 2))

def smape(pred, true):
    m = true != 0
    return np.mean(2*np.abs(true[m] - pred[m]) / (np.abs(true[m]) + np.abs(pred[m])))

# --- GRAPHE (animation) : propagation de l'onde réelle vs prédite + erreur ---
# U n'est rempli qu'aux multiples de ndt pendant le rollout -> on anime seulement
# ces pas, sinon la courbe prédite retomberait à zéro.
frames = np.arange(0, Nt + 1, ndt)

fig_anim, (axA, axB) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ligne_reel, = axA.plot([], [], "r",   lw=2, label="réel")
ligne_pred, = axA.plot([], [], "b--", lw=2, label="prédit")
ymax = np.abs(U_reel[:, nodes]).max() * 1.2
axA.set_xlim(0, L); axA.set_ylim(-ymax, ymax)
axA.set_ylabel("u"); axA.legend(loc="upper right"); axA.grid(True)
axA.set_title(f"Propagation de l'onde — {VARIANT_NAME}")

ligne_err, = axB.plot([], [], "k", lw=1.5, label="|prédit - réel|")
err_max = max(np.max([np.abs(U[m, nodes] - U_reel[m, nodes]).max() for m in frames]) * 1.2, 1e-9)
axB.set_xlim(0, L); axB.set_ylim(0, err_max)
axB.set_xlabel("x"); axB.set_ylabel("erreur absolue"); axB.legend(loc="upper right"); axB.grid(True)

titre = fig_anim.suptitle("")

def maj(m):
    ligne_reel.set_data(x, U_reel[m, nodes])
    ligne_pred.set_data(x, U[m, nodes])
    ligne_err.set_data(x, np.abs(U[m, nodes] - U_reel[m, nodes]))
    titre.set_text(f"t = {m*dt:.3f}  (pas {m})")
    return ligne_reel, ligne_pred, ligne_err, titre

anim = animation.FuncAnimation(fig_anim, maj, frames=frames, interval=50, blit=False)
anim.save(OUTPUT_DIR / "propagation_onde.gif", writer="pillow", fps=20, dpi=110)
plt.close(fig_anim)
print(f"Animation sauvegardée : {OUTPUT_DIR / 'propagation_onde.gif'}")

# --- Erreurs par pas : L2 relative, Linf, RMSE, sMAPE ---
steps  = np.arange(2*ndt, Nt + 1, ndt)          # indices réellement remplis
t_axis = steps * dt
l2_list    = [l2_rel(U[k, nodes], U_reel[k, nodes])          for k in steps]
linf_list  = [np.max(np.abs(U[k, nodes] - U_reel[k, nodes])) for k in steps]
rmse_list  = [rmse(U[k, nodes], U_reel[k, nodes])             for k in steps]
smape_list = [100.0 * smape(U[k, nodes], U_reel[k, nodes])    for k in steps]

# --- GRAPHE : erreur (L2 relative et Linf) en fonction du temps ---
plt.figure(figsize=(9, 5))
plt.plot(t_axis, l2_list,   "o-", ms=3, label="erreur L2 relative")
plt.plot(t_axis, linf_list, "s-", ms=3, label="erreur max absolue (Linf)")
plt.yscale("log")
plt.xlabel("t"); plt.ylabel("erreur"); plt.grid(True, which="both"); plt.legend()
plt.title("Erreur du rollout en fonction du temps")
plt.savefig(OUTPUT_DIR / "erreur_temps.png", dpi=150, bbox_inches="tight")
plt.close()

# --- GRAPHE : RMSE en fonction du temps (métrique utilisée pour l'ablation) ---
plt.figure(figsize=(9, 5))
plt.plot(t_axis, rmse_list, "o-", ms=3, label="RMSE")
plt.yscale("log")
plt.xlabel("t"); plt.ylabel("RMSE"); plt.grid(True, which="both"); plt.legend()
plt.title(f"RMSE du rollout en fonction du temps ({VARIANT_NAME})")
plt.savefig(OUTPUT_DIR / "rmse_temps.png", dpi=150, bbox_inches="tight")
plt.close()

# --- GRAPHE : sMAPE en fonction du temps ---
plt.figure(figsize=(9, 5))
plt.plot(t_axis, smape_list, "s-", ms=3, label="sMAPE")
plt.xlabel("t"); plt.ylabel("erreur (%)"); plt.grid(True); plt.legend()
plt.title("sMAPE du rollout en fonction du temps")
plt.savefig(OUTPUT_DIR / "smape_temps.png", dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# SEUIL DE DIVERGENCE : premier pas où RMSE > DIVERGENCE_THRESHOLD_FRAC * A
# ============================================================
divergence_rmse_threshold = DIVERGENCE_THRESHOLD_FRAC * A
divergence_idx = next((i for i, e in enumerate(rmse_list) if e > divergence_rmse_threshold), None)
divergence_step = int(steps[divergence_idx]) if divergence_idx is not None else None
divergence_time = float(t_axis[divergence_idx]) if divergence_idx is not None else None
if divergence_step is not None:
    print(f"Divergence : RMSE > {divergence_rmse_threshold:.4g} au pas {divergence_step} (t={divergence_time:.3f})")
else:
    print(f"Pas de divergence détectée (RMSE toujours <= {divergence_rmse_threshold:.4g} sur tout le rollout)")

# ============================================================
# SPECTRE FFT DE L'ERREUR à un pas de comparaison fixe (COMPARISON_STEP)
# ============================================================
cmp_idx        = int(np.argmin(np.abs(steps - COMPARISON_STEP)))
cmp_step_reel  = int(steps[cmp_idx])
error_field    = U[cmp_step_reel, nodes] - U_reel[cmp_step_reel, nodes]

fft_vals  = np.fft.rfft(error_field)
fft_freqs = np.fft.rfftfreq(len(error_field), d=dx)
fft_power = np.abs(fft_vals) ** 2
f_nyquist = 0.5 / dx
hf_mask   = fft_freqs > 0.5 * f_nyquist          # moitié haute du spectre spatial
hf_energy_frac = float(fft_power[hf_mask].sum() / (fft_power.sum() + 1e-30))

plt.figure(figsize=(9, 5))
plt.plot(fft_freqs, fft_power, "-", lw=1.2)
plt.axvline(0.5 * f_nyquist, color="k", ls="--", lw=1, label="0.5 x Nyquist")
plt.yscale("log")
plt.xlabel("fréquence spatiale (1/m)"); plt.ylabel("puissance |FFT(erreur)|²")
plt.title(f"Spectre de l'erreur au pas {cmp_step_reel} (t={cmp_step_reel*dt:.3f}) — {VARIANT_NAME}\n"
          f"énergie haute fréquence (> 0.5 Nyquist) = {100*hf_energy_frac:.1f}%")
plt.grid(True, which="both"); plt.legend()
plt.savefig(OUTPUT_DIR / "spectre_erreur.png", dpi=150, bbox_inches="tight")
plt.close()

# =====================================================
# EXPORT METRICS.JSON : source de données pour Comparatifs/compare_methodes.py
# =====================================================
l2_final,    l2_max    = l2_list[-1],    max(l2_list)
linf_final,  linf_max  = linf_list[-1],  max(linf_list)
rmse_final,  rmse_max  = rmse_list[-1],  max(rmse_list)
smape_final, smape_max = smape_list[-1], max(smape_list)

metrics = {
    "variant": VARIANT_NAME,
    "A": float(A), "omega": float(omega),
    "steps": [int(s) for s in steps],
    "t_axis": [float(t) for t in t_axis],
    "l2_rel": [float(v) for v in l2_list],
    "linf": [float(v) for v in linf_list],
    "rmse": [float(v) for v in rmse_list],
    "smape": [float(v) for v in smape_list],
    "l2_final": float(l2_final), "l2_max": float(l2_max),
    "linf_final": float(linf_final), "linf_max": float(linf_max),
    "rmse_final": float(rmse_final), "rmse_max": float(rmse_max),
    "smape_final": float(smape_final), "smape_max": float(smape_max),
    "divergence_threshold_frac": DIVERGENCE_THRESHOLD_FRAC,
    "divergence_rmse_threshold": float(divergence_rmse_threshold),
    "divergence_step": divergence_step,
    "divergence_time": divergence_time,
    "comparison_step": cmp_step_reel,
    "fft_freqs": [float(v) for v in fft_freqs],
    "fft_power": [float(v) for v in fft_power],
    "hf_energy_frac_at_comparison_step": hf_energy_frac,
    "train_time_s": float(train_time_s),
    "n_params": int(n_params),
    "best_val_loss": float(meilleure_val),
}
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metriques exportees : {OUTPUT_DIR / 'metrics.json'}")

# =====================================================
# RESUME : valeurs phares -> outputs_<variant>/resume.txt
# =====================================================
with open(OUTPUT_DIR / "resume.txt", "w") as f:
    f.write("=====  RESUME DU RUN  =====\n\n")

    f.write(f"--- Variante : {VARIANT_NAME} ---\n\n")

    f.write("--- Configuration ---\n")
    f.write(f"Grille          : Nt={Nt}, Nx={Nx}, SS={SS}, ndt={ndt}\n")
    f.write(f"Rollout (A,w)   : A={A}, omega={omega}\n")
    f.write(f"Dataset         : {len(df):,} lignes\n")
    f.write(f"Features        : {len(INPUTS)} entrees, {len(OUTPUTS)} sortie(s)\n")
    for s in ["train", "val", "test"]:
        n = (df["split"] == s).sum()
        f.write(f"  split {s:5s}   : {n:>8,} lignes ({100*n/len(df):.1f} %)\n")
    f.write(f"Parametres NN   : {n_params:,}\n\n")

    f.write("--- Entrainement ---\n")
    f.write(f"Temps entrainement : {train_time_s:.2f} s\n")
    f.write(f"Val minimale    : {meilleure_val:.6e}\n\n")

    f.write("--- Temps d'execution ---\n")
    f.write(f"Simulation reelle (FD) : {time_phys:.6f} s\n")
    f.write(f"Rollout predit (NN)    : {time_pred:.6f} s\n")
    f.write(f"Ratio NN / FD          : {time_pred/time_phys:.2f}\n\n")

    f.write("--- Erreurs du rollout ---\n")
    f.write(f"L2 relative  : finale = {l2_final:.4e}  |  max = {l2_max:.4e}\n")
    f.write(f"Linf absolue : finale = {linf_final:.4e}  |  max = {linf_max:.4e}\n")
    f.write(f"RMSE         : finale = {rmse_final:.4e}  |  max = {rmse_max:.4e}\n")
    f.write(f"sMAPE (%)    : finale = {smape_final:.3f}  |  max = {smape_max:.3f}\n\n")

    f.write("--- Divergence & spectre (ablation hautes frequences) ---\n")
    f.write(f"Seuil de divergence : RMSE > {DIVERGENCE_THRESHOLD_FRAC} * A = {divergence_rmse_threshold:.4e}\n")
    if divergence_step is not None:
        f.write(f"Pas de divergence   : n={divergence_step}  (t={divergence_time:.3f})\n")
    else:
        f.write("Pas de divergence   : aucune (RMSE reste sous le seuil sur tout le rollout)\n")
    f.write(f"Pas de comparaison FFT : n={cmp_step_reel}  (t={cmp_step_reel*dt:.3f})\n")
    f.write(f"Energie haute frequence (> 0.5 Nyquist) au pas de comparaison : {100*hf_energy_frac:.2f}%\n")

print(f"Resume sauvegarde : {OUTPUT_DIR / 'resume.txt'}")