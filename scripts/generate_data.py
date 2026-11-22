"""Génère un jeu de données SYNTHÉTIQUE et documenté.

ATTENTION : ces données servent uniquement à tester le système.
Elles ne permettent PAS de démontrer la performance réelle d'un modèle de crédit.

Règle de génération (connue, donc vérifiable) : le risque de défaut augmente
avec le taux d'effort (mensualité / revenu), et diminue avec l'âge
et l'ancienneté de l'activité. Un bruit aléatoire est ajouté.

Usage : python scripts/generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

GRAINE = 42
N = 5000
SORTIE = Path(__file__).resolve().parent.parent / "data" / "credit_synthetique.csv"


def generer(n: int = N, graine: int = GRAINE) -> pd.DataFrame:
    rng = np.random.default_rng(graine)
    age = rng.integers(18, 76, n)
    revenu = np.round(rng.lognormal(mean=12.3, sigma=0.6, size=n), -3).clip(30_000, 5_000_000)
    duree = rng.choice([3, 6, 9, 12, 18, 24, 36, 48, 60], size=n)
    montant = np.round(revenu * rng.uniform(0.5, 12, n), -4).clip(50_000, 50_000_000)
    anciennete = np.round(rng.uniform(0, np.minimum(age - 17, 40)), 1)

    taux_effort = (montant / duree) / revenu
    score = (
        -2.2 + 2.5 * taux_effort - 0.025 * (age - 40) - 0.08 * anciennete + rng.normal(0, 0.6, n)
    )
    proba = 1 / (1 + np.exp(-score))
    defaut = (rng.uniform(size=n) < proba).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "revenu_mensuel": revenu,
            "montant_credit": montant,
            "duree_mois": duree,
            "anciennete_activite": anciennete,
            "defaut": defaut,
        }
    )


if __name__ == "__main__":
    df = generer()
    SORTIE.parent.mkdir(exist_ok=True)
    df.to_csv(SORTIE, index=False)
    print(f"{len(df)} lignes écrites dans {SORTIE}")
    print(f"Taux de défaut : {df['defaut'].mean():.1%}")
