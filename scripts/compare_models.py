"""Compare une version candidate du modèle à la version de référence (Partie V).

Usage :
  python scripts/compare_models.py                       -> v1 (référence) vs v2
  python scripts/compare_models.py --candidat model_v1   -> contrôle à blanc

Produit reports/non_regression.md et reports/non_regression.json.
Les tolérances viennent de app/config.py : elles sont fixées AVANT la comparaison.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, recall_score

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from app import config  # noqa: E402
from scripts.train import calculer_metriques, charger_split  # noqa: E402

MODELES = RACINE / "models"


def intervalle_bootstrap(y, pred_ref, pred_cand, metrique, n=2000, graine=0):
    """IC à 95 % de l'écart (candidat - référence), tirages appariés sur le même jeu de test."""
    if n == 0:
        return None  # bootstrap désactivé (tests rapides en CI)
    rng = np.random.default_rng(graine)
    ecarts = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        ecarts.append(metrique(y[i], pred_cand[i]) - metrique(y[i], pred_ref[i]))
    bas, haut = np.percentile(ecarts, [2.5, 97.5])
    return round(float(bas), 4), round(float(haut), 4)


def comparer(nom_ref: str = "model_v1", nom_cand: str = "model_v2", n_bootstrap=2000) -> dict:
    _, X_test, _, y_test = charger_split()
    y = y_test.to_numpy()
    ref = joblib.load(MODELES / f"{nom_ref}.joblib")
    cand = joblib.load(MODELES / f"{nom_cand}.joblib")

    m_ref = calculer_metriques(ref, X_test, y_test)
    m_cand = calculer_metriques(cand, X_test, y_test)
    d_rappel = round(m_cand["rappel"] - m_ref["rappel"], 4)
    d_f1 = round(m_cand["f1"] - m_ref["f1"], 4)

    pred_ref = (ref.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
    pred_cand = (cand.predict_proba(X_test)[:, 1] >= 0.5).astype(int)

    criteres = {
        "rappel": {
            "ecart": d_rappel,
            "tolerance": -config.TOLERANCE_RAPPEL,
            "ic95": intervalle_bootstrap(y, pred_ref, pred_cand, recall_score, n_bootstrap),
            "conforme": d_rappel >= -config.TOLERANCE_RAPPEL,
        },
        "f1": {
            "ecart": d_f1,
            "tolerance": -config.TOLERANCE_F1,
            "ic95": intervalle_bootstrap(y, pred_ref, pred_cand, f1_score, n_bootstrap),
            "conforme": d_f1 >= -config.TOLERANCE_F1,
        },
    }
    decision = "ACCEPTEE" if all(c["conforme"] for c in criteres.values()) else "REFUSEE"
    return {
        "reference": nom_ref,
        "candidat": nom_cand,
        "taille_test": int(len(y)),
        "defauts_test": int(y.sum()),
        "metriques_reference": m_ref,
        "metriques_candidat": m_cand,
        "criteres": criteres,
        "decision": decision,
    }


def ecrire_rapport(res: dict) -> None:
    lignes = [
        f"# Rapport de non-régression : {res['candidat']} vs {res['reference']}",
        "",
        f"Jeu de test : {res['taille_test']} dossiers dont {res['defauts_test']} défauts "
        "(découpage fixe, graine 42). Seuil de décision : 0,5 pour les deux versions.",
        "",
        "| Métrique | Référence | Candidat | Écart |",
        "|---|---|---|---|",
    ]
    for k, v in res["metriques_reference"].items():
        c = res["metriques_candidat"][k]
        lignes.append(f"| {k} | {v} | {c} | {c - v:+.4f} |")
    lignes += [
        "",
        "| Critère | Écart | Tolérance | IC 95 % (bootstrap) | Verdict |",
        "|---|---|---|---|---|",
    ]
    for k, c in res["criteres"].items():
        verdict = "conforme" if c["conforme"] else "RÉGRESSION"
        lignes.append(
            f"| {k} | {c['ecart']:+.4f} | ≥ {c['tolerance']:+.2f} | {c['ic95']} | {verdict} |"
        )
    lignes += ["", f"**Décision : {res['decision']}**", ""]

    dossier = RACINE / "reports"
    dossier.mkdir(exist_ok=True)
    (dossier / "non_regression.md").write_text("\n".join(lignes), encoding="utf-8")
    (dossier / "non_regression.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n".join(lignes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="model_v1")
    parser.add_argument("--candidat", default="model_v2")
    args = parser.parse_args()
    ecrire_rapport(comparer(args.reference, args.candidat))
