"""Configuration centrale : variables, bornes de validation, seuils et chemins.

Toutes les règles du contrat sont définies ICI, une seule fois.
Les tests importent ces constantes : si une borne change, les tests suivent.
"""

import os
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent

# Ordre des variables attendu par le modèle (ne pas changer sans réentraîner)
FEATURES = [
    "age",
    "revenu_mensuel",
    "montant_credit",
    "duree_mois",
    "anciennete_activite",
]

# Bornes du contrat (voir rapport, section 2)
AGE_MIN, AGE_MAX = 18, 75
REVENU_MAX = 100_000_000  # FCFA, borne basse : strictement > 0
MONTANT_MAX = 500_000_000  # FCFA, borne basse : strictement > 0
DUREE_MIN, DUREE_MAX = 1, 120  # mois
ANCIENNETE_MIN, ANCIENNETE_MAX = 0, 60  # années

# Seuils de traduction probabilité -> niveau de risque
SEUIL_MOYEN = 0.30
SEUIL_ELEVE = 0.60

# Règle métier (correction ANO-02) : au-delà, le dossier est signalé pour revue
# Taux d'effort = mensualité / revenu mensuel (seuil usuel de crédit responsable)
SEUIL_TAUX_EFFORT = 0.40

# Tolérances de non-régression (EX07, EX12) - fixées AVANT toute comparaison
TOLERANCE_RAPPEL = 0.02  # baisse absolue maximale du rappel
TOLERANCE_F1 = 0.02  # baisse absolue maximale du F1
TOLERANCE_PROBA = 1e-9  # écart max de probabilité après sauvegarde/rechargement

# Modèle versionné utilisé par l'API (surchargeable par variable d'environnement)
MODEL_PATH = Path(os.getenv("MODEL_PATH", RACINE / "models" / "model_v1.joblib"))
