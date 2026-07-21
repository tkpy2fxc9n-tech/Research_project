# ============================================================
# IMPORTS (tout regroupé ici)
# ============================================================
from pathlib import Path
import time

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

# Dossier où sont sauvegardés tous les graphes
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# PARAMÈTRES PHYSIQUES ET GRILLE
# ============================================================
L   = 1

Nt   = 500
Nx   = 100
SS   = 11
Ntot = Nx + 2*SS

nodes = np.arange(SS, Ntot-SS)
i_left, i_right = SS, Ntot - SS

t_end = 5
dt = t_end / Nt
dx = L / (Nx - 1)
t_axis_abs = np.arange(Nt + 1) * dt   # instants absolus 0, dt, 2dt, ..., t_end

# E et rho ne sont PLUS fixes : ils varient par simulation pour que le modele
# generalise a "n'importe quelle poutre" (rigidite/densite quelconques), pas
# seulement E=1, rho=2 comme avant. c_onde = sqrt(E/rho) est donc ajoute comme
# feature scalaire du reseau (cf. COLONNES plus bas), au meme titre que
# duree_requete. Nx/L/dx restent fixes (cf. discussion : seuls E/rho varient).
E_RANGE   = (0.5, 2.0)
RHO_RANGE = (0.5, 2.0)
CFL_MAX   = 0.9   # marge de securite sous la limite de stabilite CFL<=1

def tirer_E_rho(rng):
    # Tirage par rejet : redessine tant que le CFL resultant (dt/dx*sqrt(E/rho))
    # ne respecte pas CFL_MAX, pour garantir la stabilite du schema explicite
    # quelle que soit la combinaison (E, rho) tiree dans E_RANGE x RHO_RANGE.
    while True:
        E_i, rho_i = rng.uniform(*E_RANGE), rng.uniform(*RHO_RANGE)
        cfl_i = dt / dx * np.sqrt(E_i / rho_i)
        if cfl_i <= CFL_MAX:
            return E_i, rho_i, cfl_i

# Nombre de simulations FD generees pour le dataset, chacune avec une CL
# aleatoire DIFFERENTE (cf. cl_aleatoire ci-dessous) ET une poutre (E, rho)
# DIFFERENTE. Remplace l'ancienne grille (A, omega) : le but n'est plus de
# parametrer la CL par (A, omega) sur une poutre fixe, mais d'exposer le
# reseau a une famille de CL et de poutres suffisamment large pour qu'il
# generalise ensuite a une CL et une poutre choisies librement.
N_SIMULATIONS = 40

# Parametres de la famille aleatoire de CL (somme de "bosses" gaussiennes) :
CL_N_MODES_MAX = 4              # nombre max de bosses sommees pour une CL
CL_AMPL_RANGE  = (0.005, 0.1)   # amplitude (valeur absolue) de chaque bosse
CL_SIGMA_RANGE = (0.05, 0.3)    # largeur (ecart-type, en secondes) de chaque bosse
CL_T0_MARGE    = 0.3            # marge (s) evitant de centrer une bosse trop pres de t=0 ou t_end
rng_cl = np.random.default_rng(SEED)


def u_right_val(A, omega, t):
    # Impulsion gaussienne "historique" a 2 parametres. Ne sert plus a generer
    # le dataset (remplacee par cl_aleatoire, plus diverse) mais reste utile
    # pour construire a la main une CL de test/demo precise et reproductible.
    sigma = np.interp(omega, [1.0, 10.0], [0.15, 0.07])
    t0    = 4.0 * sigma
    return A * np.exp(-((t - t0) / sigma) ** 2)


def cl_aleatoire(rng, t_axis):
    # Genere une CL "boite noire" : somme d'un nombre aleatoire de bosses
    # gaussiennes d'amplitude/largeur/position/signe aleatoires. Sert a
    # exposer le reseau, pendant l'entrainement, a une famille de formes bien
    # plus large qu'une impulsion unique a 2 parametres — condition
    # necessaire pour qu'il generalise ensuite a une CL choisie librement.
    signal = np.zeros_like(t_axis)
    n_modes = rng.integers(1, CL_N_MODES_MAX + 1)
    for _ in range(n_modes):
        amp   = rng.uniform(*CL_AMPL_RANGE) * rng.choice([-1.0, 1.0])
        t0    = rng.uniform(CL_T0_MARGE, t_end - CL_T0_MARGE)
        sigma = rng.uniform(*CL_SIGMA_RANGE)
        signal = signal + amp * np.exp(-((t_axis - t0) / sigma) ** 2)
    return signal


def simuler_fd(cl_signal, cfl):
    # Différences finies (leapfrog) de l'équation d'onde 1D. CI nulles, bord
    # gauche encastré, bord droit piloté par cl_signal : tableau (Nt+1,)
    # donnant la valeur imposee au pas k (k=0..Nt), quelle que soit son
    # origine (cl_aleatoire, u_right_val, ou un vecteur ecrit a la main).
    # cfl = dt/dx*sqrt(E/rho) de LA poutre simulee (E, rho variables, cf.
    # tirer_E_rho) — plus une constante globale unique.
    u   = np.zeros(Ntot)
    u_1 = np.zeros(Ntot)

    u_storage    = np.zeros((Nt + 1, Ntot))
    u_storage[0] = u.copy()

    for n in range(Nt):
        u_new = np.zeros(Ntot)
        u_new[i_left:i_right+1] = (2.0 * u[i_left:i_right+1] - u_1[i_left:i_right+1]
                                    + cfl**2 * (u[i_left-1:i_right] - 2.0 * u[i_left:i_right+1] + u[i_left+1:i_right+2]))
        u_new[:i_left+1] = 0.0
        u_new[i_right:]  = cl_signal[n + 1]

        u_1 = u.copy()
        u   = u_new
        u_storage[n + 1] = u.copy()

    return u_storage


def calcul_ut(u_storage):
    # Derivee temporelle du champ complet (differences finies centrees ; aux
    # deux bords temporels — n=0 et n=Nt — on n'a qu'un voisin donc schema
    # decentre avant/arriere). Meme forme que u_storage : (Nt+1, Ntot).
    ut = np.empty_like(u_storage)
    ut[1:-1] = (u_storage[2:] - u_storage[:-2]) / (2.0 * dt)
    ut[0]    = (u_storage[1]  - u_storage[0])   / dt
    ut[-1]   = (u_storage[-1] - u_storage[-2])  / dt
    return ut


def calcul_uxx(u_storage):
    # Derivee seconde spatiale (courbure), evaluee directement sur les noeuds
    # physiques de la poutre. Les 2 voisins (nodes-1, nodes+1) existent
    # toujours grace au padding SS : pour le 1er/dernier noeud ce sont les
    # valeurs de bord (encastrement=0 ou deplacement impose), pas du padding
    # arbitraire. Retourne (Nt+1, Nx), deja restreint aux noeuds physiques.
    return (u_storage[:, nodes - 1] - 2.0 * u_storage[:, nodes] + u_storage[:, nodes + 1]) / dx**2


# ============================================================
# GÉNÉRATION DES DONNÉES — paradigme "stepper d'etat"
# Une ligne = (etat complet de la poutre a l'instant I=n_start : U, Ut, Uxx sur
# TOUS les noeuds + fenetre des CL RELATIVE a I sur la duree demandee D,
# remplie sur [0,D] et nulle au-dela + duree demandee D*dt)
#   -> delta_u sur TOUTE la poutre entre I et I+D.
#
# Point cle : la fenetre de CL est indexee en pas RELATIFS depuis I (position
# 0 = valeur de la CL a l'instant I, position m = valeur a I+m), pas en pas
# ABSOLUS de la simulation. Deux lignes avec le meme D mais des I differents
# produisent donc le meme "type" de vecteur (meme nombre de valeurs utiles
# suivies de zeros) : c'est ce qui permet au modele d'apprendre une notion de
# D independante de la position I sur la trajectoire, et donc de pouvoir
# quantifier ensuite un "D limite" de precision/stabilite qui soit une
# propriete de D seul (cf. objectif : (etat a I, CL sur D) -> etat a I+D).
# ============================================================
PAS_ETAT  = 5     # stride entre instants de depart I=n_start echantillonnes sur la trajectoire
PAS_DUREE = 5     # stride entre durees D explorees pour chaque I
NDT_MAX   = Nt    # plus grand D (en pas) couvert par la fenetre de CL ; borne aussi n_query <= n_start+NDT_MAX
# NB : reduire NDT_MAX limite la duree D maximale que le modele pourra jamais
# recevoir (utile pour une premiere exploration rapide et moins couteuse) ;
# NDT_MAX=Nt couvre tout l'intervalle possible (I=0 -> D peut aller jusqu'a Nt).
# NB dataset : le nombre de lignes croit en O((Nt/PAS_ETAT) x (Nt/PAS_DUREE))
# par simulation. Baisser PAS_ETAT/PAS_DUREE (jusqu'a 1) donne le dataset le
# plus riche mais le plus gros ; les augmenter l'allege. A ajuster selon la
# RAM/temps de calcul dispo.

COLS_U   = [f"u_j{j:03d}"        for j in range(Nx)]        # champ deplacement U a l'instant I
COLS_UT  = [f"ut_j{j:03d}"       for j in range(Nx)]        # champ vitesse Ut = dU/dt a l'instant I
COLS_UXX = [f"uxx_j{j:03d}"      for j in range(Nx)]        # champ courbure Uxx = d2U/dx2 a l'instant I
COLS_CL  = [f"cl_dm{m:03d}"      for m in range(NDT_MAX + 1)]  # fenetre des CL relative a I : m = pas depuis I
COLS_OUT = [f"delta_u_j{j:03d}"  for j in range(Nx)]

COLONNES = (["sim_idx", "E", "rho", "n_start", "n_query"]
            + COLS_U + COLS_UT + COLS_UXX + COLS_CL
            + ["duree_requete", "c_onde"] + COLS_OUT)

lignes_dataset = []

for sim_idx in range(N_SIMULATIONS):

    E_i, rho_i, cfl_i = tirer_E_rho(rng_cl)                     # poutre (rigidite/densite) de cette simulation
    c_onde_i   = np.sqrt(E_i / rho_i)
    cl_signal  = cl_aleatoire(rng_cl, t_axis_abs)               # CL de cette simulation, connue a l'avance
    u_storage  = simuler_fd(cl_signal, cfl_i)
    ut_storage = calcul_ut(u_storage)
    uxx_nodes  = calcul_uxx(u_storage)                          # (Nt+1, Nx), deja restreint aux noeuds physiques

    U_all  = u_storage[:, nodes]                                # (Nt+1, Nx)
    Ut_all = ut_storage[:, nodes]                                # (Nt+1, Nx)

    for n_start in range(0, Nt, PAS_ETAT):

        U0, Ut0, Uxx0 = U_all[n_start], Ut_all[n_start], uxx_nodes[n_start]
        n_query_max = min(n_start + NDT_MAX, Nt)

        for n_query in range(n_start + PAS_DUREE, n_query_max + 1, PAS_DUREE):

            D = n_query - n_start                                 # duree demandee, en pas
            hist_cl = np.zeros(NDT_MAX + 1)
            hist_cl[:D + 1] = cl_signal[n_start:n_query + 1]       # fenetre relative [I, I+D], indexee depuis 0

            duree_requete = D * dt
            delta_u = U_all[n_query] - U0

            ligne = np.concatenate((
                [sim_idx, E_i, rho_i, n_start, n_query],
                U0, Ut0, Uxx0,
                hist_cl,
                [duree_requete, c_onde_i],
                delta_u,
            ))
            lignes_dataset.append(ligne)

df = pd.DataFrame(lignes_dataset, columns=COLONNES)
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

OUTPUTS = COLS_OUT
meta    = ["sim_idx", "E", "rho", "n_start", "n_query", "split"]
INPUTS  = [c for c in df.columns if c not in meta + OUTPUTS]

train_mask = df["split"] == "train"
norm_stats = pd.DataFrame({
    "mean": df.loc[train_mask, INPUTS + OUTPUTS].mean(),
    "std" : df.loc[train_mask, INPUTS + OUTPUTS].std(),
})
norm_stats["std"] = norm_stats["std"].replace(0, 1)

# NB : les colonnes u_j*/ut_j*/uxx_j* ne valent 0 que sur les lignes ou
# n_start=0 (repos initial) ; des que n_start>0 elles varient reellement
# d'une ligne a l'autre (contrairement a l'ancienne version ou u0 etait fige
# a 0 partout). Le garde-fou std->1 ci-dessus reste utile en cas de colonne
# degenerescente residuelle mais ne devrait plus s'appliquer systematiquement
# a ces colonnes.
# Les colonnes cl_dm* d'indice m proche de NDT_MAX ne sont non nulles que sur
# les lignes ou D >= m (donc les requetes a grande duree) -> std faible sur
# ces colonnes -> z-score peut y amplifier le signal. Effet de bord du
# remplissage causal de la fenetre, a surveiller si NDT_MAX est pris grand.

for col in INPUTS + OUTPUTS:
    df[col + "_n"] = (df[col] - norm_stats.loc[col, "mean"]) / norm_stats.loc[col, "std"]

# ============================================================
# HYPERPARAMÈTRES
# ============================================================
HIDDEN_SIZES = [512, 256, 128]

LEARNING_RATE = 1e-3
N_EPOCHS      = 800
BATCH_SIZE    = 512
LAMBDA_SMOOTH = 0.1   # poids de la penalite de lissage spatial dans la loss (§6bis) ; regularisation legere

# Lissage post-hoc du champ reconstruit a l'inference (predire_etat_poutre). La
# penalite ci-dessus (LAMBDA_SMOOTH) reduit un peu le bruit nœud-a-nœud mais ne
# suffit pas seule : ici on applique un lissage Laplacien ITERE (contrairement a
# l'ancien rollout qui l'appliquait une fois par bloc mais de facon cumulee sur
# des dizaines de blocs, ici il n'y a qu'UNE seule requete donc il faut repeter
# le lissage plusieurs fois pour un effet comparable). alpha proche de 0.25
# (limite de stabilite du schema) et quelques dizaines d'iterations.
SMOOTH_ALPHA = 0.24
SMOOTH_ITERS = 20

print(f"Dataset : {len(df):,} lignes")
print(f"Splits  : {df['split'].value_counts().to_dict()}")

# ============================================================
# TENSEURS PyTorch + DataLoader d'entraînement
# ============================================================
INPUTS_N  = [f"{x}_n" for x in INPUTS]
OUTPUTS_N = [f"{x}_n" for x in OUTPUTS]

X_train = df.loc[df["split"] == "train", INPUTS_N].values.astype(np.float32)
y_train = df.loc[df["split"] == "train", OUTPUTS_N].values.astype(np.float32)

X_val = df.loc[df["split"] == "val", INPUTS_N].values.astype(np.float32)
y_val = df.loc[df["split"] == "val", OUTPUTS_N].values.astype(np.float32)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                          batch_size=BATCH_SIZE, shuffle=True)

# Moyennes / ecarts-types, dans l'ordre EXACT de INPUTS / OUTPUTS — utiles pour
# normaliser/denormaliser une ligne construite "a la main" a l'inference.
# Restent en float64 : les colonnes cl_k* loin de l'impulsion (queue de la
# gaussienne) ont une moyenne/ecart-type d'une magnitude si petite (~1e-200 et
# moins) qu'un cast en float32 avant normalisation les ferait sous-deborder a
# exactement 0.0 -> 0/0 = NaN. Le pipeline d'entrainement evite deja ce piege
# (il normalise les colonnes du DataFrame en float64 et ne caste en float32
# qu'apres, via X_train.astype(np.float32)) ; on reproduit la meme logique ici.
mu_in  = norm_stats.loc[INPUTS,  "mean"].values
sd_in  = norm_stats.loc[INPUTS,  "std" ].values
mu_out = norm_stats.loc[OUTPUTS, "mean"].values
sd_out = norm_stats.loc[OUTPUTS, "std" ].values

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

# ============================================================
# ENTRAÎNEMENT
# Pas de rollout auto-regressif ici (chaque ligne est une requete independante)
# -> pas besoin de pushforward, de lissage Laplacien ni de bruit d'entree pour
# stabiliser une consommation recursive des sorties, contrairement au paradigme
# pointwise precedent.
#
# Penalite de lissage spatial : rien dans l'architecture (MLP dense, sorties
# independantes) ni dans la MSE ne pousse deux noeuds voisins a avoir des
# valeurs proches, contrairement a l'ancien stencil translation-invariant qui
# assurait cette coherence implicitement. On l'ajoute donc explicitement.
# ============================================================
def smoothness_loss(pred):
    diffs = pred[:, 1:] - pred[:, :-1]
    return (diffs ** 2).mean()

historique_train  = []
historique_val    = []
historique_smooth = []
meilleure_val     = float("inf")

for epoch in range(1, N_EPOCHS + 1):

    modele.train()
    perte_train  = 0.0
    perte_smooth = 0.0
    for X_batch, y_batch in train_loader:
        optimiseur.zero_grad()
        prediction  = modele(X_batch)
        perte_data  = criterion(prediction, y_batch)
        perte_liss  = smoothness_loss(prediction)
        total_loss  = perte_data + LAMBDA_SMOOTH * perte_liss
        total_loss.backward()
        optimiseur.step()
        perte_train  += perte_data.item()
        perte_smooth += perte_liss.item()
    perte_train  /= len(train_loader)
    perte_smooth /= len(train_loader)

    modele.eval()
    with torch.no_grad():
        pred_val = modele(torch.tensor(X_val)).numpy()
    perte_val = ((pred_val - y_val)**2).mean()
    scheduler.step(perte_val)

    historique_train.append(perte_train)
    historique_val  .append(perte_val)
    historique_smooth.append(perte_smooth)

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d}/{N_EPOCHS}  —  train: {perte_train:.4f}  |  val: {perte_val:.4f}  |  lissage: {perte_smooth:.4f}")

    if perte_val < meilleure_val:
        meilleure_val = perte_val
        torch.save(modele.state_dict(), SCRIPT_DIR / "model.pth")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(historique_train,  label="Train")
ax.plot(historique_val,    label="Validation")
ax.plot(historique_smooth, label="Lissage")
ax.set_xlabel("Epoch"); ax.set_ylabel("Erreur MSE")
ax.set_title("Courbe d'apprentissage")
ax.set_yscale("log"); ax.legend(); ax.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "courbe_apprentissage.png", dpi=150, bbox_inches="tight")
plt.close()

modele.load_state_dict(torch.load(SCRIPT_DIR / "model.pth", weights_only=True))
print(f"Meilleur modèle rechargé — val minimale : {meilleure_val:.6f}")

# ============================================================
# ÉVALUATION TEACHER-FORCING (sur le jeu de test, entrées propres)
# ============================================================
df_test = df[df["split"] == "test"].reset_index(drop=True)

X_new    = df_test[INPUTS_N].values.astype(np.float32)
y_true_n = df_test[OUTPUTS_N].values
y_true   = df_test[OUTPUTS].values
duree_test = df_test["duree_requete"].values

modele.eval()
with torch.no_grad():
    y_pred_n = modele(torch.tensor(X_new)).numpy()

y_pred = np.zeros_like(y_pred_n)
for i, col in enumerate(OUTPUTS):
    y_pred[:, i] = y_pred_n[:, i] * norm_stats.loc[col, "std"] + norm_stats.loc[col, "mean"]

# --- GRAPHE : predit vs reel, un panneau par tiers de duree demandee ---
tiers_bornes = [0, t_end/3, 2*t_end/3, t_end]
tiers_labels = ["courte", "moyenne", "longue"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (ax, lab) in enumerate(zip(axes, tiers_labels)):
    masque = (duree_test >= tiers_bornes[i]) & (duree_test < tiers_bornes[i+1] + 1e-9)
    y_r, y_p = y_true[masque].ravel(), y_pred[masque].ravel()
    y_rn, y_pn = y_true_n[masque].ravel(), y_pred_n[masque].ravel()

    if len(y_r) > 5000:
        idx = np.random.default_rng(0).choice(len(y_r), 5000, replace=False)
        y_r_plot, y_p_plot = y_r[idx], y_p[idx]
    else:
        y_r_plot, y_p_plot = y_r, y_p

    ax.scatter(y_r_plot, y_p_plot, alpha=0.4, s=8)
    lim = max(abs(y_r_plot).max(), abs(y_p_plot).max(), 1e-9)
    ax.plot([-lim, lim], [-lim, lim], "r--", lw=1, label="prédiction parfaite")

    mse_norm = ((y_pn - y_rn) ** 2).mean()
    r2 = 1 - mse_norm / y_rn.var()
    ax.set_xlabel("delta_u réel"); ax.set_ylabel("delta_u prédit")
    ax.set_title(f"Durée {lab} (∈[{tiers_bornes[i]:.2f}, {tiers_bornes[i+1]:.2f}]s)\nMSE (norm)={mse_norm:.2e}  |  R²={r2:.3f}")
    ax.legend(); ax.grid(True)

fig.suptitle("Test sur toutes les données test du dataset — par tiers de durée demandée", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "test_predictions.png", dpi=150, bbox_inches="tight")
plt.close()

for i, lab in enumerate(tiers_labels):
    masque = (duree_test >= tiers_bornes[i]) & (duree_test < tiers_bornes[i+1] + 1e-9)
    mse_norm = ((y_pred_n[masque] - y_true_n[masque]) ** 2).mean()
    r2 = 1 - mse_norm / y_true_n[masque].var()
    print(f"Durée {lab:8s} : MSE (norm) = {mse_norm:.4e}  |  R² = {r2:.4f}")

# --- GRAPHE : erreur moyenne (test set) en fonction de la position sur la poutre ---
x_nodes = np.linspace(0, L, Nx)
erreur_abs_par_noeud = np.abs(y_pred - y_true).mean(axis=0)

plt.figure(figsize=(9, 5))
plt.plot(x_nodes, erreur_abs_par_noeud, "o-", ms=3)
plt.xlabel("x"); plt.ylabel("erreur absolue moyenne (delta_u)")
plt.title("Erreur moyenne (test set) en fonction de la position sur la poutre")
plt.grid(True)
plt.savefig(OUTPUT_DIR / "erreur_par_position.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# INFÉRENCE DIRECTE — une seule passe forward par requête. Le rollout n'est
# plus seulement possible mais NATIF au paradigme : U0/Ut0/Uxx0 + n_start
# peuvent decrire n'importe quel etat de la poutre, pas seulement le repos a
# t=0 (cf. section GENERATION DES DONNEES ci-dessus).
#
# cl_signal : tableau indexe en pas ABSOLUS (comme celui genere par
# cl_aleatoire ou u_right_val), couvrant au moins jusqu'a n_query. C'est
# volontairement une CL QUELCONQUE, pas seulement celle vue a l'entrainement
# — l'appelant est libre d'y mettre ce qu'il veut (un vecteur ecrit a la
# main, une autre forme, etc.), tant que sa longueur/duree D=n_query-n_start
# ne depasse pas NDT_MAX (au-dela, hors du domaine d'entrainement).
#
# c_onde = sqrt(E/rho) de LA poutre consideree : necessaire car (U0,Ut0,Uxx0,
# CL,D) ne suffit pas a determiner l'evolution si la rigidite/densite de la
# poutre change (cf. u_tt = c_onde^2 * u_xx) — l'appelant peut y mettre
# n'importe quelle poutre (pas seulement celles vues a l'entrainement).
# ============================================================
def construire_entree(U0, Ut0, Uxx0, cl_signal, c_onde, n_start, n_query):
    n_query = np.atleast_1d(n_query)
    n_start = np.full(len(n_query), n_start) if np.isscalar(n_start) else np.atleast_1d(n_start)
    X = np.zeros((len(n_query), len(INPUTS)))   # float64 : cf. remarque sur mu_in/sd_in ci-dessus
    for i, (ns, nq) in enumerate(zip(n_start, n_query)):
        D = nq - ns                             # duree demandee, en pas (cf. NDT_MAX)
        hist_cl = np.zeros(NDT_MAX + 1)
        hist_cl[:D + 1] = cl_signal[ns:nq + 1]  # fenetre relative a n_start, indexee depuis 0
        duree_requete = D * dt
        X[i] = np.concatenate((U0, Ut0, Uxx0, hist_cl, [duree_requete, c_onde]))
    X_norm = (X - mu_in) / sd_in   # normalisation en float64
    return X_norm.astype(np.float32)   # cast en float32 seulement apres, comme X_train


def predire_etat_poutre(U0, Ut0, Uxx0, cl_signal, c_onde, n_start, n_query):
    entree_scalaire = np.isscalar(n_query)
    n_query_arr = np.atleast_1d(n_query)
    X_norm = construire_entree(U0, Ut0, Uxx0, cl_signal, c_onde, n_start, n_query_arr)

    with torch.no_grad():
        delta_pred_n = modele(torch.tensor(X_norm)).numpy()
    delta_pred = delta_pred_n * sd_out + mu_out

    champs = np.zeros((len(n_query_arr), Ntot))
    for i, nq in enumerate(n_query_arr):
        champs[i, nodes]     = U0 + delta_pred[i]
        champs[i, :i_left+1] = 0.0                              # CL reimposees en dur, comme aujourd'hui
        champs[i, i_right:]  = cl_signal[nq]

        j0, j1 = i_left + 1, i_right                            # nœuds interieurs (hors bords)
        for _ in range(SMOOTH_ITERS):
            lap = champs[i, j0-1:j1-1] - 2*champs[i, j0:j1] + champs[i, j0+1:j1+1]
            champs[i, j0:j1] += SMOOTH_ALPHA * lap

    return champs[0] if entree_scalaire else champs


# ============================================================
# DÉMONSTRATION — comparaison réel vs prédit sur une simulation
# ============================================================
x = np.linspace(0, L, Nx)

cl_signal_demo = cl_aleatoire(rng_cl, t_axis_abs)   # une CL "comme a l'entrainement" (meme famille aleatoire)
E_demo, rho_demo, cfl_demo = tirer_E_rho(rng_cl)    # une poutre "comme a l'entrainement" (meme plage E/rho)
c_onde_demo = np.sqrt(E_demo / rho_demo)

t0 = time.perf_counter()
U_reel = simuler_fd(cl_signal_demo, cfl_demo)
time_phys = time.perf_counter() - t0

Ut_reel  = calcul_ut(U_reel)
Uxx_reel = calcul_uxx(U_reel)                                    # (Nt+1, Nx), deja aux noeuds physiques

# Etat de depart de la demo principale : repos a t=0 (n_start=0), comme dans
# l'ancienne version. Le modele reconstruit toute la trajectoire en une
# seule passe forward batchee (une ligne par n_query). Borne a NDT_MAX si
# celui-ci est plus petit que Nt (D ne peut pas depasser NDT_MAX).
n_start_demo = 0
U0_demo   = U_reel[n_start_demo, nodes]
Ut0_demo  = Ut_reel[n_start_demo, nodes]
Uxx0_demo = Uxx_reel[n_start_demo]

t0 = time.perf_counter()
# NB : suppose NDT_MAX >= Nt (valeur par defaut) pour reconstruire TOUTE la
# trajectoire depuis t=0 ; si NDT_MAX < Nt, seules les lignes 0..NDT_MAX sont
# predites, le reste de U_predit reste nul (a adapter si NDT_MAX est reduit).
n_query_tous = np.arange(n_start_demo + 1, min(n_start_demo + NDT_MAX, Nt) + 1)
U_predit = np.zeros((Nt + 1, Ntot))
U_predit[n_start_demo] = U_reel[n_start_demo]
U_predit[n_start_demo + 1:n_query_tous[-1] + 1] = predire_etat_poutre(
    U0_demo, Ut0_demo, Uxx0_demo, cl_signal_demo, c_onde_demo, n_start_demo, n_query_tous
)
time_pred = time.perf_counter() - t0

print("temps physique (simulation FD complète) :", round(time_phys, 6))
print("temps prédit (toutes les durées, 1 batch):", round(time_pred, 6))

# --- Validation supplementaire : prediction depuis un etat INTERMEDIAIRE
# (n_start != 0), pour verifier que le modele generalise a n'importe quel
# etat de depart et pas seulement au repos initial (c'est tout le sens du
# changement de paradigme demande : U/Ut/Uxx en entree, pas juste A/omega). ---
n_start_mid = Nt // 2
U0_mid, Ut0_mid, Uxx0_mid = U_reel[n_start_mid, nodes], Ut_reel[n_start_mid, nodes], Uxx_reel[n_start_mid]
n_query_mid  = np.arange(n_start_mid + 1, min(n_start_mid + NDT_MAX, Nt) + 1)
U_predit_mid = predire_etat_poutre(U0_mid, Ut0_mid, Uxx0_mid, cl_signal_demo, c_onde_demo, n_start_mid, n_query_mid)
erreur_mid   = np.abs(U_predit_mid[:, nodes] - U_reel[n_start_mid + 1:n_query_mid[-1] + 1, nodes])
print(f"Validation depuis un etat intermediaire (n_start={n_start_mid}) : "
      f"erreur abs moyenne = {erreur_mid.mean():.4e}  |  max = {erreur_mid.max():.4e}")

# --- Validation supplementaire : CL ET POUTRE choisies A LA MAIN (ici
# l'ancienne impulsion u_right_val(A,omega,t) sur une poutre E_perso/rho_perso
# precise, pas tiree au hasard), pour verifier concretement que le reseau
# accepte une CL ET une poutre quelconques fournies par l'utilisateur, pas
# seulement celles vues en entrainement — c'est tout le sens de la derniere
# demande. E_perso/rho_perso restent dans E_RANGE/RHO_RANGE : au-dela, ce
# serait de l'extrapolation hors du domaine d'entrainement, a tester a part. ---
A_perso, omega_perso = 0.05, 6.0
cl_signal_perso = u_right_val(A_perso, omega_perso, t_axis_abs)
E_perso, rho_perso = 0.8, 1.0
cfl_perso    = dt / dx * np.sqrt(E_perso / rho_perso)
c_onde_perso = np.sqrt(E_perso / rho_perso)

U_reel_perso    = simuler_fd(cl_signal_perso, cfl_perso)
Ut_reel_perso   = calcul_ut(U_reel_perso)
Uxx_reel_perso  = calcul_uxx(U_reel_perso)

n_start_perso = 0
U0_perso, Ut0_perso, Uxx0_perso = (U_reel_perso[n_start_perso, nodes],
                                   Ut_reel_perso[n_start_perso, nodes],
                                   Uxx_reel_perso[n_start_perso])
n_query_perso  = np.arange(n_start_perso + 1, min(n_start_perso + NDT_MAX, Nt) + 1)
U_predit_perso = predire_etat_poutre(U0_perso, Ut0_perso, Uxx0_perso, cl_signal_perso, c_onde_perso,
                                      n_start_perso, n_query_perso)
erreur_perso   = np.abs(U_predit_perso[:, nodes] - U_reel_perso[n_start_perso + 1:n_query_perso[-1] + 1, nodes])
print(f"Validation avec une CL et une poutre choisies a la main "
      f"(E={E_perso}, rho={rho_perso}) : "
      f"erreur abs moyenne = {erreur_perso.mean():.4e}  |  max = {erreur_perso.max():.4e}")


# ============================================================
# GRAPHES : comparaison réel vs prédit (inférence directe, sans rollout)
# ============================================================
def l2_rel(pred, true, eps=1e-12):
    return np.linalg.norm(pred - true) / (np.linalg.norm(true) + eps)

def smape(pred, true):
    m = true != 0
    if not m.any():           # poutre exactement au repos (debut de trajectoire) -> rien a mesurer
        return 0.0
    return np.mean(2*np.abs(true[m] - pred[m]) / (np.abs(true[m]) + np.abs(pred[m])))

# --- GRAPHE (animation) : propagation de l'onde réelle vs prédite + erreur ---
PAS_ANIMATION = 5   # cadence du GIF (cosmetique) — l'inference elle-meme peut etre interrogee a n'importe quel pas
frames = np.arange(0, Nt + 1, PAS_ANIMATION)

fig_anim, (axA, axB) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

ligne_reel, = axA.plot([], [], "r",   lw=2, label="réel")
ligne_pred, = axA.plot([], [], "b--", lw=2, label="prédit")
ymax = np.abs(U_reel[:, nodes]).max() * 1.2
axA.set_xlim(0, L); axA.set_ylim(-ymax, ymax)
axA.set_ylabel("u"); axA.legend(loc="upper right"); axA.grid(True)

ligne_err, = axB.plot([], [], "k", lw=1.5, label="|prédit - réel|")
err_max = max(np.abs(U_predit[:, nodes] - U_reel[:, nodes]).max() * 1.2, 1e-9)
axB.set_xlim(0, L); axB.set_ylim(0, err_max)
axB.set_xlabel("x"); axB.set_ylabel("erreur absolue"); axB.legend(loc="upper right"); axB.grid(True)

titre = fig_anim.suptitle("")

def maj(m):
    ligne_reel.set_data(x, U_reel[m, nodes])
    ligne_pred.set_data(x, U_predit[m, nodes])
    ligne_err.set_data(x, np.abs(U_predit[m, nodes] - U_reel[m, nodes]))
    titre.set_text(f"Poutre entière, inférence directe — t = {m*dt:.3f}  (pas {m})")
    return ligne_reel, ligne_pred, ligne_err, titre

anim = animation.FuncAnimation(fig_anim, maj, frames=frames, interval=50, blit=False)
anim.save(OUTPUT_DIR / "propagation_onde.gif", writer="pillow", fps=20, dpi=110)
plt.close(fig_anim)
print(f"Animation sauvegardée : {OUTPUT_DIR / 'propagation_onde.gif'}")

# --- GRAPHE : erreur (L2 relative et Linf) en fonction de la DURÉE DEMANDÉE ---
# Chaque point part de la verite terrain (etat initial reel, historique de CL
# reel) : contrairement au rollout auto-regressif d'avant, l'erreur ne
# s'accumule plus d'un pas a l'autre. Une eventuelle croissance ici reflete la
# difficulte intrinseque de predire loin depuis une description compressee,
# pas une composition d'erreurs.
steps  = np.arange(1, Nt + 1)
t_axis = steps * dt
l2_list   = [l2_rel(U_predit[k, nodes], U_reel[k, nodes])         for k in steps]
linf_list = [np.max(np.abs(U_predit[k, nodes] - U_reel[k, nodes])) for k in steps]

plt.figure(figsize=(9, 5))
plt.plot(t_axis, l2_list,   "o-", ms=3, label="erreur L2 relative")
plt.plot(t_axis, linf_list, "s-", ms=3, label="erreur max absolue (Linf)")
plt.yscale("log")
plt.xlabel("durée demandée (t)"); plt.ylabel("erreur"); plt.grid(True, which="both"); plt.legend()
plt.title("Erreur de l'inférence directe en fonction de la durée demandée\n(pas de rollout : aucune accumulation d'un pas à l'autre)")
plt.savefig(OUTPUT_DIR / "erreur_vs_duree.png", dpi=150, bbox_inches="tight")
plt.close()

smape_list = [100.0 * smape(U_predit[k, nodes], U_reel[k, nodes]) for k in steps]

plt.figure(figsize=(9, 5))
plt.plot(t_axis, smape_list, "s-", ms=3, label="sMAPE")
plt.xlabel("durée demandée (t)"); plt.ylabel("erreur (%)"); plt.grid(True); plt.legend()
plt.title("sMAPE de l'inférence directe en fonction de la durée demandée")
plt.savefig(OUTPUT_DIR / "smape_vs_duree.png", dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# BENCHMARK TEMPS/FLOPs — FD (trajectoire complète) vs NN (une seule requête)
# ============================================================
torch.set_num_threads(1)   # comparaison reproductible

def chrono(fonction, n_repeat=15, n_warmup=3):
    for _ in range(n_warmup):
        fonction()
    duree = np.array([(lambda: (time.perf_counter(), fonction(), time.perf_counter())[::2])() for _ in range(n_repeat)])
    d = duree[:, 1] - duree[:, 0]
    return d.mean(), d.std(), np.median(d)

def fd_once():
    return simuler_fd(cl_signal_demo, cfl_demo)

def nn_once():
    return predire_etat_poutre(U0_demo, Ut0_demo, Uxx0_demo, cl_signal_demo, c_onde_demo,
                                n_start_demo, min(n_start_demo + NDT_MAX, Nt))

_, _, med_fd = chrono(fd_once)
_, s_nn, med_nn = chrono(nn_once)
print(f"FD (trajectoire complète jusqu'à t_end) : {med_fd*1e3:7.3f} ms")
print(f"NN (une seule requête, coût constant)   : {med_nn*1e3:7.3f} ms  (±{s_nn*1e3:.3f})")
print(f"speedup FD/NN = {med_fd/med_nn:.2f}x   (>1 = le réseau est plus rapide ; le FD est linéaire en durée demandée, le NN reste constant)")

from torch.utils.flop_counter import FlopCounterMode
with FlopCounterMode(display=False) as fc:
    modele(torch.zeros((1, len(INPUTS_N))))
print(f"FLOPs réseau (une requête) ≈ {fc.get_total_flops():,.0f}")
print(f"FLOPs FD (~Nt*Nx*8)        ≈ {Nt*Nx*8:,}")


# =====================================================
# RESUME : valeurs phares -> outputs/resume.txt
# =====================================================
l2_final,    l2_max    = l2_list[-1],    max(l2_list)
linf_final,  linf_max  = linf_list[-1],  max(linf_list)
smape_final, smape_max = smape_list[-1], max(smape_list)

with open(OUTPUT_DIR / "resume.txt", "w") as f:
    f.write("=====  RESUME DU RUN  =====\n\n")

    f.write("--- Paradigme ---\n")
    f.write("Stepper d'etat : (U,Ut,Uxx sur toute la poutre a I=n_start, fenetre des CL\n")
    f.write("relative a I sur la duree demandee D, D, c_onde=sqrt(E/rho) de la poutre)\n")
    f.write("-> delta_u sur la poutre entiere a I+D.\n")
    f.write("CL d'entrainement tirees aleatoirement (cl_aleatoire, famille large de sommes de\n")
    f.write("bosses gaussiennes) plutot qu'une seule impulsion a 2 parametres, et poutre (E,rho)\n")
    f.write("tiree aleatoirement dans E_RANGE x RHO_RANGE a chaque simulation, pour permettre au\n")
    f.write("modele d'accepter ensuite une CL et une poutre quelconques choisies par l'utilisateur.\n")
    f.write("Une seule passe forward par requete ; n_start balaie toute la trajectoire (pas\n")
    f.write("seulement le repos initial) -> rollout natif possible en reinjectant l'etat predit.\n\n")

    f.write("--- Configuration ---\n")
    f.write(f"Grille          : Nt={Nt}, Nx={Nx}, SS={SS}\n")
    f.write(f"Poutres         : E in {E_RANGE}, rho in {RHO_RANGE} (tirage par rejet, CFL<={CFL_MAX})\n")
    f.write(f"Dataset         : {len(df):,} lignes ({N_SIMULATIONS} simulations (CL+poutre aleatoires) x n_start (pas {PAS_ETAT}) x n_query (pas {PAS_DUREE}))\n")
    f.write(f"Features        : {len(INPUTS)} entrees, {len(OUTPUTS)} sorties\n")
    for s in ["train", "val", "test"]:
        n = (df["split"] == s).sum()
        f.write(f"  split {s:5s}   : {n:>8,} lignes ({100*n/len(df):.1f} %)\n")
    f.write(f"Parametres NN   : {n_params:,}\n\n")

    f.write("--- Entrainement ---\n")
    f.write(f"Val minimale    : {meilleure_val:.6e}\n")
    for i, lab in enumerate(tiers_labels):
        masque = (duree_test >= tiers_bornes[i]) & (duree_test < tiers_bornes[i+1] + 1e-9)
        mse_norm = ((y_pred_n[masque] - y_true_n[masque]) ** 2).mean()
        r2 = 1 - mse_norm / y_true_n[masque].var()
        f.write(f"Duree {lab:8s} : MSE (norm) = {mse_norm:.4e} | R2 = {r2:.4f}\n")
    f.write("\n")

    f.write("--- Temps d'execution ---\n")
    f.write(f"FD, trajectoire complete jusqu'a t_end : {med_fd:.6f} s\n")
    f.write(f"NN, une seule requete (cout constant)  : {med_nn:.6f} s\n")
    f.write(f"speedup FD/NN                          : {med_fd/med_nn:.2f}\n\n")

    f.write("--- Erreur de l'inference directe en fonction de la duree demandee ---\n")
    f.write("(pas d'accumulation : chaque requete part de la verite terrain)\n")
    f.write(f"L2 relative  : finale = {l2_final:.4e}  |  max = {l2_max:.4e}\n")
    f.write(f"Linf absolue : finale = {linf_final:.4e}  |  max = {linf_max:.4e}\n")
    f.write(f"sMAPE (%)    : finale = {smape_final:.3f}  |  max = {smape_max:.3f}\n")

print(f"Resume sauvegarde : {OUTPUT_DIR / 'resume.txt'}")
