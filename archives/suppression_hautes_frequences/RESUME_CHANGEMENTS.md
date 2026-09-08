# Résumé — ablation "suppression des hautes fréquences"

Fait pendant que tu dormais, à relire à tête reposée. Rien n'a été exécuté
(hors un test du script d'agrégation avec des données factices, dans mon
scratchpad, jamais dans ce dossier — voir tout en bas). Le fichier original
`(Uxx, Ut)=>(delta_u) and ndt=5 et N et M.py` n'a pas été touché.

## Fichiers créés

```
Code_suppression_hautes_frequences/
├── (Uxx, Ut)=>(delta_u) and ndt=5 et N et M.py                 <- ORIGINAL, intact
├── (Uxx, Ut)=>(delta_u) and ndt=5 et N et M_v0_baseline.py     <- référence pour la comparaison
├── (Uxx, Ut)=>(delta_u) and ndt=5 et N et M_v1_smoothing.py
├── (Uxx, Ut)=>(delta_u) and ndt=5 et N et M_v2_noise_injection.py
├── (Uxx, Ut)=>(delta_u) and ndt=5 et N et M_v3_spectral_penalty.py
├── (Uxx, Ut)=>(delta_u) and ndt=5 et N et M_v4_spectral_filter.py
├── (Uxx, Ut)=>(delta_u) and ndt=5 et N et M_v5_pushforward.py
├── RESUME_CHANGEMENTS.md                                        <- ce fichier
└── Comparatifs/
    └── compare_methodes.py     <- à lancer APRÈS les 6 scripts ci-dessus
```

Chaque script `_vX_*.py` est autonome (regénère ses données, entraîne son
propre modèle, fait son propre rollout) — comme demandé. J'ai ajouté une
copie `v0_baseline` en plus des 5 demandées : c'est le fichier original tel
quel (mêmes hyperparamètres, aucune méthode ajoutée), avec seulement les
corrections d'infrastructure ci-dessous, pour servir de point de comparaison
"baseline actuelle" dans le tableau final. Sans elle il n'y avait rien à
mettre dans la colonne baseline du tableau comparatif (le script original ne
peut pas fonctionner tel quel — voir bug ci-dessous).

## Bug corrigé dans les 6 copies (pas dans l'original)

En fin de script, l'original écrit dans
`Code_comparaison_des_inputs/Comparaisons/comparative_table.xlsx` (colonne
`Ut_Uxx_PF_Lap`). Deux problèmes :
1. Ce fichier appartient à une **autre étude d'ablation** (comparaison des
   champs d'entrée U / Ut / Uxx), pas à celle-ci — y écrire aurait pollué
   ses données.
2. Le chemin utilisé (`SCRIPT_DIR.parent / "Comparaisons"`) pointait vers
   `Research_Individual_Project/Comparaisons/`, qui n'existe pas depuis
   l'emplacement actuel du script — **ça aurait fait planter le run avant
   même d'écrire `resume.txt`**.

Je n'ai pas touché à l'original (comme demandé), mais dans les 6 copies j'ai
remplacé ce bloc par un export local (`metrics.json`, voir plus bas) lu par
`Comparatifs/compare_methodes.py`.

## Changements communs aux 6 copies (infrastructure, pas des méthodes)

- Chaque copie écrit dans son propre `outputs_<variante>/` (au lieu du
  dossier `outputs/` partagé de l'original) — sinon les 6 scripts
  s'écraseraient mutuellement (graphes, `model.pth`, `resume.txt`).
- Ajout du **RMSE par pas** (`sqrt(mean((pred-true)**2))`), en plus des
  métriques déjà présentes (L2 relative, Linf, sMAPE).
- Ajout d'un **seuil de divergence** : premier pas où RMSE dépasse
  `DIVERGENCE_THRESHOLD_FRAC * A` (constante en haut du script, défaut
  `0.5`, soit la moitié de l'amplitude d'entrée du run). Je n'ai trouvé
  aucun seuil de ce type ailleurs dans ton projet (cherché dans
  `Code_comparaison_des_inputs`, `Code`, `Old Code`) donc j'en ai défini un
  — à ajuster si besoin. Comme la courbe RMSE(t) complète est sauvegardée
  dans `metrics.json`, tu peux recalculer le seuil de divergence a
  posteriori sans relancer les runs.
- Ajout d'un **spectre FFT de l'erreur** à un pas de comparaison fixe
  (`COMPARISON_STEP = 300`, ~60% du rollout, identique pour les 6 runs
  puisque `A` est déterministe). Énergie haute fréquence = puissance au-delà
  de 0.5×Nyquist / puissance totale.
- Chaque script exporte `outputs_<variante>/metrics.json` (courbes
  complètes + métriques résumées + spectre) — c'est ce que lit
  `Comparatifs/compare_methodes.py`.
- **Retiré des 6 copies** (suite à ta demande de nettoyage) tout ce qui ne
  sert pas à cette comparaison précise : le graphe de vérification EDP
  (u_tt vs u_xx, réel et prédit), l'éval teacher-forcing sur le jeu de test
  (`test_predictions.png` + son bloc de calcul), et le benchmark de vitesse
  FD vs NN (qui refaisait tourner une simulation FD ET un rollout complets
  ~18 fois chacun juste pour mesurer un temps — le plus gros poste de calcul
  inutile). Ces blocs existaient dans le script original mais ne
  contribuent à aucune des 3 métriques demandées (RMSE(t), pas de
  divergence, spectre FFT de l'erreur). `n_params` et `train_time_s` sont
  gardés (mesure directe, gratuite) ; les 3 champs qui dépendaient du
  benchmark (`rollout_time_median_s`, `rollout_time_std_s`,
  `speedup_fd_over_nn`) ont disparu de `metrics.json` et du tableau
  comparatif en conséquence.
- **Remise ensuite** (tu voulais voir visuellement l'effet de chaque
  méthode) : l'animation `propagation_onde.gif`, avec ses deux panneaux —
  onde réelle vs prédite en haut, erreur absolue |prédit - réel| en bas —
  identique à l'original, juste retitrée avec le nom de la variante.

## Méthode par variante

### v0_baseline
Aucun changement de méthode. `NOISE_STD=0.10`, `SMOOTH_ALPHA=0.20`,
`LAMBDA_PF=1.0` — les valeurs déjà présentes dans ton fichier original.

### v1_smoothing — Laplacian smoothing renforcé
- **Bug de cohérence corrigé** : le lissage Laplacien existait déjà dans la
  boucle de rollout principale, mais **pas** dans `reconstruct()` — la
  fonction qui construit la "fausse" prédiction auto-régressive utilisée par
  le pushforward pendant l'entraînement. Le réseau était donc entraîné à
  corriger des entrées non lissées, alors qu'au vrai rollout il ne voit que
  des champs déjà lissés. J'ai ajouté le même bloc de lissage dans
  `reconstruct()`.
- `SMOOTH_ALPHA` : 0.20 → **0.35**. Le schéma
  `u[j] += alpha*(u[j-1]-2u[j]+u[j+1])` est une moyenne pondérée stable tant
  que `alpha <= 0.5` ; 0.35 reste dans la zone stable.

### v2_noise_injection — bruit ciblé haute fréquence
L'original injecte déjà du bruit gaussien à l'entraînement
(`NOISE_STD=0.10`), mais c'est un bruit **blanc isotrope** (i.i.d. sur les
132 features) : son spectre est plat, il ne cible pas spécifiquement le mode
damier. J'ai **remplacé la forme du bruit**, en gardant la même amplitude
(0.10), par un bruit concentré exactement sur le mode de Nyquist spatial :
```
pattern(k) = (-1)^k                    sur les voisins k = -SS..+SS
bruit(échantillon, feature) = eps * pattern(feature)
eps ~ N(0, NOISE_STD²), tiré une fois par échantillon
```
Le pattern est dupliqué pour (u_dot, u_xx) puis répété sur les 3 lags
temporels, dans le même ordre de colonnes que le dataset. Le test isole donc
la question : *"bruiter le mode damier spécifiquement, à amplitude égale,
fait-il mieux que le bruit blanc déjà en place ?"*

### v3_spectral_penalty — pénalité de norme spectrale (1ère couche)
Ajoutée **en plus** de la baseline (pas de remplacement). On pénalise le
gain de la 1ère couche du MLP sur la direction "damier" :
```
c = direction damier unitaire (même construction que v2, normalisée ||c||=1)
g = W1 @ c                      # gain de chaque neurone caché au mode damier pur
loss += LAMBDA_SPEC * mean(g²)  # LAMBDA_SPEC = 1e-3
```
`LAMBDA_SPEC=1e-3` est choisi petit devant la loss de données pour agir
comme régularisation douce — à monter si l'effet est trop faible.

### v4_spectral_filter — filtre passe-bas FFT (remplace le lissage)
Remplace le lissage Laplacien (mis à `SMOOTH_ALPHA=0`) par un filtrage net
en fréquence : FFT du champ intérieur → mise à zéro des fréquences
au-delà de `CUTOFF_FRAC * Nyquist` (défaut 0.5) → IFFT. Appliqué à chaque
bloc du rollout **et** dans `reconstruct()` (même correction de cohérence
qu'en v1, pour que pushforward et rollout voient la même chose).

### v5_pushforward — pushforward renforcé
Vérifié : le pushforward était déjà actif et câblé (`pushforward_loss()`,
`reconstruct()`, `PF_SAMPLES`, poids qui monte progressivement de 0 à
`LAMBDA_PF` sur 2 époques). Seul changement : `LAMBDA_PF` 1.0 → **3.0**.
Rien d'autre modifié (`PF_WARMUP` inchangé, `reconstruct()` non retouché)
pour n'isoler que l'effet du poids.

## Comment lancer chaque script

Chaque fichier est autonome : il régénère les données, entraîne son modèle
et fait son rollout en un seul run. Commande générique :
```bash
python "(Uxx, Ut)=>(delta_u) and ndt=5 et N et M_v1_smoothing.py"
```
(idem pour v0, v2, v3, v4, v5 — indépendants les uns des autres, tu peux les
lancer dans n'importe quel ordre ou en parallèle sur le HPC).

**Dépendances** : j'ai trouvé un environnement conda `wave_env` sur cette
machine qui a déjà tout (`numpy`, `pandas`, `torch`, `matplotlib`,
`openpyxl`) — c'est probablement celui que tu utilises déjà pour ces
scripts. Sur le HPC, adapte selon ton module/env habituel.

**Sorties de chaque script** (dans `outputs_<variante>/`, créé
automatiquement) :
- `resume.txt` — résumé lisible (config, erreurs, seuil de divergence, etc.)
- `metrics.json` — toutes les courbes + métriques, lu par l'agrégation
- `courbe_apprentissage.png`, `erreur_temps.png`, `rmse_temps.png`,
  `smape_temps.png`, `spectre_erreur.png`, `propagation_onde.gif` (réel vs
  prédit + erreur absolue, pour juger visuellement l'effet de chaque méthode
  sur le damier)
- `model.pth` — poids du meilleur modèle

**Une fois les 6 scripts terminés** (ou même seulement quelques-uns — les
variantes manquantes sont juste ignorées avec un avertissement), lance
l'agrégation :
```bash
cd Comparatifs
python compare_methodes.py
```
Ça lit tous les `outputs_<variante>/metrics.json` du dossier parent et
produit :
- `Comparatifs/comparative_table.xlsx` — feuille "Resume" (1 ligne/métrique,
  1 colonne/méthode) + feuille "RolloutCurves" (courbes RMSE(t) par méthode)
- `Comparatifs/outputs/*.png` — RMSE(t) comparé (log + linéaire), pas de
  divergence par méthode, spectres d'erreur superposés, énergie haute
  fréquence par méthode, RMSE final vs max

## Hyperparamètres inchangés (volontairement)

`N_EPOCHS=20`, taille du dataset (`N=6` amplitudes × `N=6` pulsations),
architecture (`[64,32,16]`), seed (`0`) : identiques dans les 6 copies, pour
que la comparaison isole bien l'effet de chaque méthode et pas une
différence de budget d'entraînement. Comme tu as dit que le temps de calcul
n'est pas un problème sur le HPC, tu peux augmenter `N_EPOCHS` ou la
densité du dataset (`N`) si tu veux des résultats plus poussés — mais fais-le
de façon identique dans les 6 fichiers pour garder la comparaison valide, je
ne l'ai pas fait moi-même pour ne pas m'écarter de "mêmes hyperparamètres
sauf mention contraire".

## Vérifications faites (sans exécuter les 6 scripts, comme demandé)

- Les 7 fichiers (original + 6 copies) compilent (`py_compile`, vérifie la
  syntaxe uniquement).
- Grep croisé : `VARIANT_NAME`, `SMOOTH_ALPHA`, `LAMBDA_PF` ont bien la
  valeur attendue dans chaque fichier ; les mécanismes spécifiques
  (`checker_mask`/`checker_dir`, `spectral_lowpass`, `spectral_hf_penalty`)
  n'apparaissent que dans les fichiers concernés ; `reconstruct()` n'est
  patché que dans v1 et v4 ; aucune référence résiduelle au bloc xlsx cassé.
- Le script `Comparatifs/compare_methodes.py` a été testé de bout en bout
  avec des métriques factices générées dans mon scratchpad (jamais dans ce
  dossier) : il produit bien le xlsx et les 5 PNG sans erreur. Ce test a été
  entièrement nettoyé après coup, aucune trace dans `Comparatifs/`.
- Je n'ai PAS exécuté les 6 scripts d'ablation eux-mêmes (dataset réel +
  entraînement), comme demandé.
- Après le nettoyage (retrait des blocs EDP/animation/teacher-forcing/
  benchmark) : les 6 fichiers recompilent, grep de non-régression refait sur
  les mêmes points (paramètres par variante, `reconstruct()` patché
  seulement en v1/v4, aucune référence résiduelle aux variables supprimées
  `y_pred_n`, `med_fd`, `chrono`, etc.), et `Comparatifs/compare_methodes.py`
  re-testé de bout en bout avec des `metrics.json` factices respectant le
  nouveau schéma (sans les champs de timing retirés) — toujours nettoyé
  après coup, aucune trace laissée.
