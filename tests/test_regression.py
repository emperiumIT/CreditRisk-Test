"""Tests de non-régression (Partie V).

Principe : on compare toujours à une VERSION DE RÉFÉRENCE figée (models/model_v1),
avec des tolérances écrites AVANT la comparaison (app/config.py).

Pour tester une nouvelle version dans la CI :
  MODEL_CANDIDAT=model_v2 pytest tests/test_regression.py
"""

import json
import os

import joblib
import numpy as np
import pytest

from app import config
from scripts.compare_models import comparer
from scripts.train import calculer_metriques, charger_split

MODELES = config.RACINE / "models"
CANDIDAT = os.getenv("MODEL_CANDIDAT", "model_v1")


@pytest.fixture(scope="module")
def split():
    """Jeu de test figé (même graine) : condition d'une comparaison équitable."""
    return charger_split()


@pytest.fixture(scope="module")
def reference():
    return joblib.load(MODELES / "model_v1.joblib")


# --- 1. La référence elle-même est stable ------------------------------------
def test_metriques_reference_reproductibles(reference, split):
    """Les métriques recalculées = celles enregistrées à l'entraînement."""
    _, X_test, _, y_test = split
    enregistrees = json.loads((MODELES / "model_v1.json").read_text(encoding="utf-8"))
    recalculees = calculer_metriques(reference, X_test, y_test)
    for nom, valeur in enregistrees["metriques_test"].items():
        assert recalculees[nom] == pytest.approx(valeur, abs=1e-4), nom


# --- 2. La porte de non-régression (EX07) ------------------------------------
def test_candidat_respecte_les_tolerances():
    """Échoue si la version candidate dégrade le rappel ou le F1 au-delà de la tolérance."""
    res = comparer("model_v1", CANDIDAT, n_bootstrap=0)
    for nom, critere in res["criteres"].items():
        assert critere["conforme"], (
            f"Régression sur {nom} : écart {critere['ecart']:+.4f}, "
            f"tolérance {critere['tolerance']:+.2f}"
        )


def test_la_porte_detecte_la_regression_de_v2():
    """Preuve que la porte fonctionne : v2 (rappel -0,027, F1 -0,029) doit être refusée."""
    res = comparer("model_v1", "model_v2", n_bootstrap=0)
    assert res["decision"] == "REFUSEE"
    assert not res["criteres"]["rappel"]["conforme"]
    assert not res["criteres"]["f1"]["conforme"]


def test_la_porte_accepte_une_version_identique():
    """Contrôle à blanc : comparer la référence à elle-même ne doit rien signaler."""
    assert comparer("model_v1", "model_v1", n_bootstrap=0)["decision"] == "ACCEPTEE"


# --- 3. Cohérence après sauvegarde et rechargement (EX12) --------------------
def test_coherence_apres_sauvegarde_et_rechargement(reference, split, tmp_path):
    _, X_test, _, _ = split
    p_avant = reference.predict_proba(X_test)[:, 1]

    chemin = tmp_path / "copie.joblib"
    joblib.dump(reference, chemin)
    recharge = joblib.load(chemin)
    p_apres = recharge.predict_proba(X_test)[:, 1]

    assert np.max(np.abs(p_avant - p_apres)) <= config.TOLERANCE_PROBA


def test_predictions_deterministes(reference, split):
    _, X_test, _, _ = split
    p1 = reference.predict_proba(X_test)[:, 1]
    p2 = reference.predict_proba(X_test)[:, 1]
    assert np.array_equal(p1, p2)


# --- 4. Les sorties du pipeline restent conformes au contrat -----------------
def test_sorties_pipeline_conformes_au_contrat(client, split):
    """100 vrais dossiers du jeu de test passent par l'API : contrat respecté pour tous."""
    _, X_test, _, _ = split
    for _, ligne in X_test.head(100).iterrows():
        dossier = {
            "age": int(ligne["age"]),
            "revenu_mensuel": float(ligne["revenu_mensuel"]),
            "montant_credit": float(ligne["montant_credit"]),
            "duree_mois": int(ligne["duree_mois"]),
            "anciennete_activite": float(ligne["anciennete_activite"]),
        }
        r = client.post("/predict", json=dossier)
        assert r.status_code == 200
        corps = r.json()
        assert 0.0 <= corps["probabilite_defaut"] <= 1.0
        assert corps["niveau_risque"] in {"faible", "moyen", "eleve"}
        assert corps["version_modele"] == "1.0.0"
