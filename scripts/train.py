"""Entraîne le modèle et le sauvegarde avec ses métadonnées.

Usage :
  python scripts/train.py                          -> models/model_v1.joblib (référence)
  python scripts/train.py --version 2.0.0 --nom model_v2 --C 0.001
      -> une "nouvelle version" à comparer à la référence (Partie V)
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from app import config  # noqa: E402

DONNEES = RACINE / "data" / "credit_synthetique.csv"
GRAINE = 42


def charger_split():
    """Découpage train/test FIXE (même graine) pour comparer les versions équitablement."""
    df = pd.read_csv(DONNEES)
    X, y = df[config.FEATURES], df["defaut"]
    return train_test_split(X, y, test_size=0.2, random_state=GRAINE, stratify=y)


def calculer_metriques(modele, X_test, y_test) -> dict:
    proba = modele.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "rappel": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--nom", default="model_v1")
    parser.add_argument("--C", type=float, default=1.0, help="Régularisation")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = charger_split()
    modele = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=args.C, class_weight="balanced", max_iter=1000)),
        ]
    )
    modele.fit(X_train, y_train)
    metriques = calculer_metriques(modele, X_test, y_test)

    dossier = RACINE / "models"
    dossier.mkdir(exist_ok=True)
    joblib.dump(modele, dossier / f"{args.nom}.joblib")

    metadata = {
        "version": args.version,
        "date_entrainement": date.today().isoformat(),
        "algorithme": "LogisticRegression (class_weight=balanced) + StandardScaler",
        "hyperparametres": {"C": args.C},
        "features": config.FEATURES,
        "seuils_niveau_risque": {"moyen": config.SEUIL_MOYEN, "eleve": config.SEUIL_ELEVE},
        "donnees": "Synthétiques (generate_data.py, graine 42) - test du système uniquement",
        "domaine_entrainement": {
            f: [float(X_train[f].min()), float(X_train[f].max())] for f in config.FEATURES
        },
        "metriques_test": metriques,
    }
    (dossier / f"{args.nom}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
