# Évaluation formative n°2 — Notes validées pour le rapport de qualification

Ce document rassemble les éléments **validés** au fil du travail. Il sert de base au rapport final (Partie VIII).
Échéance : dimanche 22 novembre 2026, 23h59.

---

## 1. Contrat de l'API (règles de validation retenues)

| Champ | Type | Règle | Exemple valide |
|---|---|---|---|
| age | entier | 18 ≤ age ≤ 75 | 35 |
| revenu_mensuel | nombre | 0 < revenu ≤ 100 000 000 FCFA | 250000 |
| montant_credit | nombre | 0 < montant ≤ 500 000 000 FCFA | 1000000 |
| duree_mois | entier | 1 ≤ durée ≤ 120 | 12 |
| anciennete_activite | nombre | 0 ≤ ancienneté ≤ 60 | 4 |
| champ non prévu | — | rejeté | — |

**Décision :** la validation est **stricte** : le texte `"12"` n'est pas converti en nombre, il est rejeté. Il faut justifier ce choix dans le rapport : on préfère rejeter une donnée ambiguë plutôt que la deviner.

**Codes HTTP :** 200 = succès · 422 = données invalides · 503 = modèle indisponible · 500 = erreur interne avec un message générique, sans trace technique.

**Réponse de `/predict`** (contrat fixe, sans champ facultatif) :
```json
{ "probabilite_defaut": 0.23, "niveau_risque": "faible", "version_modele": "1.0.0" }
```
Seuils de `niveau_risque` : p < 0,30 → faible · 0,30 ≤ p < 0,60 → moyen · p ≥ 0,60 → élevé.

---

## 2. Échelle de priorité

| Priorité | Définition | Conséquence pour le pilote |
|---|---|---|
| P1 — Critique | Une défaillance mène à une décision de crédit erronée, à une faille de sécurité ou à un service inutilisable | Bloquant |
| P2 — Majeure | Dégradation de la fiabilité ou de la qualité, qu'on peut contourner | À corriger avant la mise en production |
| P3 — Mineure | Traçabilité ou confort | Recommandation |

---

## 3. Matrice des exigences (version validée)

| ID | Exigence | Résultat attendu (mesurable) | Risque principal | Priorité |
|---|---|---|---|---|
| EX01 | L'âge est compris dans l'intervalle autorisé (18–75) | Âge < 18 ou > 75 → 422 | Acceptation d'un profil incohérent ou d'un mineur | P1 |
| EX02 | Le revenu est numérique et positif (0 < r ≤ 1e8) | Texte, valeur nulle, négative ou hors plafond → 422 | Prédiction sur données aberrantes | P1 |
| EX03 | La durée est un entier valide (1–120) | Décimal, texte ou hors bornes → 422 | Calcul de risque faussé | P1 |
| EX04 | La prédiction est une probabilité valide | p fini et 0 ≤ p ≤ 1 pour toute entrée valide | Sortie non exploitable | P1 |
| EX05 | Le modèle est disponible | Modèle non chargé → 503 sur /predict ; /health indique `model_loaded: false` | Erreur technique exposée au client | P1 |
| EX06 | Une entrée incomplète est soumise à validation | Champ absent → 422 avec le nom du champ dans le message | Crash ou prédiction partielle | P1 |
| EX07 | Une nouvelle version ne dégrade pas les critères critiques | Tests fonctionnels verts ; baisse du rappel ≤ 0,02 et du F1 ≤ 0,02 par rapport à la référence | Régression silencieuse | P1 |
| EX08 | Le montant du crédit est strictement positif (0 < m ≤ 5e8) | Montant nul, négatif ou hors plafond → 422 | Incohérence métier | P1 |
| EX09 | L'ancienneté d'activité est numérique, positive ou nulle (0–60) | Valeur invalide → 422 | Mauvaise estimation du risque | P2 |
| EX10 | Les champs supplémentaires sont rejetés | Champ inconnu → 422 | Contournement du contrat d'API | P2 |
| EX11 | La réponse contient les champs attendus | `probabilite_defaut` (float), `niveau_risque` ∈ {faible, moyen, élevé}, `version_modele` | Réponse inexploitable par l'agent | P1 |
| EX12 | Les prédictions sont reproductibles | Même entrée → même sortie ; après sauvegarde et rechargement, \|Δp\| ≤ 1e-9 | Résultats instables, non auditables | P2 |
| EX13 | Les erreurs internes ne divulguent pas de traces sensibles | 500 + message générique, sans « Traceback » ni chemin de fichier ; exception journalisée côté serveur | Fuite d'information | **P1** |
| EX14 | La latence respecte l'objectif | P95 de /predict < 200 ms en local (500 requêtes) | Mauvaise expérience utilisateur | P2 |
| EX15 | Les métadonnées du modèle sont cohérentes | /model-info : version, date d'entraînement, seuils, liste des variables identique à celle du prétraitement | Traçabilité insuffisante | P3 |

---

## 4. Cas de test associés

| ID test | Exig. | Entrée (les autres champs restent valides) | Attendu |
|---|---|---|---|
| CT01-1..5 | EX01 | age = 17 / 18 / 75 / 76 / -5 | 422 / 200 / 200 / 422 / 422 |
| CT02-1..8 | EX02 | revenu = 250000 / "abc" / 0 / -1000 / 1 / 1e8 / 1e8+1 / null | 200 / 422 / 422 / 422 / 200 / 200 / 422 / 422 |
| CT03-1..8 | EX03 | durée = 12 / 1 / 120 / 0 / 121 / 12.5 / "12" / -6 | 200 / 200 / 200 / 422 / 422 / 422 / 422 / 422 |
| CT04-1 | EX04 | 1 requête valide | p fini, dans [0,1] |
| CT04-2 | EX04 | 100 profils valides aléatoires (graine fixée) | tous les p sont finis et dans [0,1] |
| CT05-1 | EX05 | modèle non chargé (simulé par un mock) | /predict → 503 ; /health → `model_loaded: false` |
| CT06-1..5 | EX06 | chacun des 5 champs retiré tour à tour | 422, avec le nom du champ dans `detail` |
| CT06-6 | EX06 | corps `{}` | 422 |
| CT06-7 | EX06 | JSON mal formé (`{"age": 35,`) | 422 |
| CT07-1 | EX07 | nouvelle version comparée à la référence | Δrappel ≥ -0,02 et ΔF1 ≥ -0,02 |
| CT08-1..5 | EX08 | montant = 0 / -1 / 1 / 5e8 / 5e8+1 | 422 / 422 / 200 / 200 / 422 |
| CT09-1..5 | EX09 | ancienneté = 0 / 60 / -1 / 61 / "x" | 200 / 200 / 422 / 422 / 422 |
| CT10-1 | EX10 | ajout de `"nom": "Paul"` | 422 |
| CT11-1 | EX11 | requête valide | 3 champs présents, avec les bons types ; niveau cohérent avec p |
| CT12-1 | EX12 | même requête envoyée 2 fois | sorties identiques |
| CT12-2 | EX12 | sauvegarde puis rechargement du modèle, sur le jeu de test | \|Δp\| ≤ 1e-9 |
| CT13-1 | EX13 | exception forcée dans la prédiction (mock) | 500, `{"detail": "Erreur interne"}`, pas de « Traceback » |
| CT14-1 | EX14 | 500 requêtes en local | P95 < 200 ms ; débit mesuré |
| CT15-1 | EX15 | GET /model-info | champs présents ; variables = celles du prétraitement |

**Précondition commune** (sauf CT05 et CT13) : API démarrée, modèle de test versionné chargé.

**Techniques de conception utilisées :** partitionnement en classes d'équivalence, analyse des valeurs limites, tests négatifs, injection de pannes par mock.
