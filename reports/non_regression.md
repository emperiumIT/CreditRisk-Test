# Rapport de non-régression : model_v2 vs model_v1

Jeu de test : 1000 dossiers dont 221 défauts (découpage fixe, graine 42). Seuil de décision : 0,5 pour les deux versions.

| Métrique | Référence | Candidat | Écart |
|---|---|---|---|
| accuracy | 0.726 | 0.702 | -0.0240 |
| precision | 0.4358 | 0.4094 | -0.0264 |
| rappel | 0.8145 | 0.7873 | -0.0272 |
| f1 | 0.5678 | 0.5387 | -0.0291 |
| roc_auc | 0.821 | 0.8092 | -0.0118 |

| Critère | Écart | Tolérance | IC 95 % (bootstrap) | Verdict |
|---|---|---|---|---|
| rappel | -0.0272 | ≥ -0.02 | (-0.0577, 0.0) | RÉGRESSION |
| f1 | -0.0291 | ≥ -0.02 | (-0.0487, -0.0101) | RÉGRESSION |

**Décision : REFUSEE**
