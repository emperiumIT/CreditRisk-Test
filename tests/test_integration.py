"""Tests d'intégration (Partie III).

Un test unitaire vérifie UN composant isolé.
Un test d'intégration vérifie que les composants COLLABORENT correctement :
validation -> prétraitement -> modèle -> API.

Technique clé : monkeypatch (fixture Pytest) remplace temporairement une
fonction ou un attribut pour PROVOQUER une panne, puis restaure tout après le test.
"""

import logging

import pandas as pd
import pytest

import app.main as main_module
from app import config
from app.prediction import ModelService, PredictionError, model_service, niveau_risque
from app.preprocessing import PreprocessingError, construire_features
from app.validation import DemandeCredit

# ============================================================================
# Scénario principal : le pipeline de bout en bout
# ============================================================================


def test_pipeline_bout_en_bout(client, payload_valide):
    # 1. Validation
    demande = DemandeCredit(**payload_valide)

    # 2. Transformation des variables : 1 ligne, 5 colonnes, dans le bon ordre
    X = construire_features(demande.model_dump())
    assert X.shape == (1, len(config.FEATURES))
    assert list(X.columns) == config.FEATURES

    # 3. Prédiction directe par le modèle
    p_directe = model_service.predict_proba(X)

    # 4. Même demande via l'API : le résultat doit être IDENTIQUE
    r = client.post("/predict", json=payload_valide)
    assert r.status_code == 200
    corps = r.json()
    assert corps["probabilite_defaut"] == pytest.approx(p_directe, abs=1e-12)

    # 5. Structure et cohérence de la réponse
    assert corps["niveau_risque"] == niveau_risque(corps["probabilite_defaut"])
    assert corps["version_modele"] == model_service.metadata["version"]


def test_metadonnees_coherentes_avec_le_modele(client):
    info = client.get("/model-info").json()
    modele = model_service.model

    # Le modèle attend exactement les variables produites par le prétraitement
    assert modele.n_features_in_ == len(config.FEATURES)
    assert list(modele.feature_names_in_) == config.FEATURES
    assert info["features"] == config.FEATURES
    # Les seuils annoncés sont ceux réellement appliqués
    seuils = {"moyen": config.SEUIL_MOYEN, "eleve": config.SEUIL_ELEVE}
    assert info["seuils_niveau_risque"] == seuils


@pytest.mark.parametrize(
    "p, attendu",
    [
        (0.0, "faible"),
        (0.2999, "faible"),
        (0.30, "moyen"),
        (0.5999, "moyen"),
        (0.60, "eleve"),
        (1.0, "eleve"),
    ],
)
def test_seuils_niveau_risque(p, attendu):
    assert niveau_risque(p) == attendu


# ============================================================================
# Scénarios complémentaires : pannes provoquées
# ============================================================================


def test_modele_impossible_a_charger(monkeypatch, payload_valide):
    """Fichier modèle introuvable au démarrage -> API en mode dégradé, pas de crash."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(config, "MODEL_PATH", config.RACINE / "models" / "inexistant.joblib")
    monkeypatch.setattr(model_service, "model", None)

    with TestClient(main_module.app) as c:
        assert c.get("/health").json() == {"status": "degraded", "model_loaded": False}
        r = c.post("/predict", json=payload_valide)
        assert r.status_code == 503
        assert c.get("/model-info").status_code == 503


def test_chargement_fichier_absent_leve_une_erreur():
    with pytest.raises(FileNotFoundError):
        ModelService().load(config.RACINE / "models" / "inexistant.joblib")


def test_variable_attendue_absente():
    donnees = {"age": 35, "revenu_mensuel": 250000}  # 3 variables manquantes
    with pytest.raises(PreprocessingError, match="manquantes"):
        construire_features(donnees)


@pytest.mark.parametrize("valeur", [float("nan"), float("inf"), "35", True])
def test_pretraitement_refuse_valeur_non_numerique_ou_non_finie(payload_valide, valeur):
    payload_valide["age"] = valeur
    with pytest.raises(PreprocessingError):
        construire_features(payload_valide)


def test_erreur_pretraitement_via_api(client, payload_valide, monkeypatch):
    """Le prétraitement échoue -> 500 générique, sans détail technique."""

    def pretraitement_en_panne(_):
        raise PreprocessingError("colonne interne XYZ introuvable")

    monkeypatch.setattr(main_module, "construire_features", pretraitement_en_panne)

    r = client.post("/predict", json=payload_valide)
    assert r.status_code == 500
    assert r.json() == {"detail": "Erreur interne"}
    assert "XYZ" not in r.text


def test_dimensions_incompatibles():
    X_4_colonnes = pd.DataFrame([[35, 250000, 1000000, 12]], columns=config.FEATURES[:4])
    with pytest.raises(PredictionError, match="Dimensions"):
        model_service.predict_proba(X_4_colonnes)


def test_dimensions_incompatibles_via_api(client, payload_valide, monkeypatch):
    monkeypatch.setattr(
        main_module,
        "construire_features",
        lambda d: pd.DataFrame([[35, 250000, 1000000, 12]], columns=config.FEATURES[:4]),
    )
    r = client.post("/predict", json=payload_valide)
    assert r.status_code == 500
    assert r.json() == {"detail": "Erreur interne"}


def test_erreur_interne_pendant_prediction(client, payload_valide, monkeypatch, caplog):
    """Exception dans le modèle -> 500 générique ; la trace va dans les LOGS, pas au client."""

    def modele_en_panne(X):
        raise RuntimeError(r"Echec lecture C:\secret\poids.bin")

    monkeypatch.setattr(model_service.model, "predict_proba", modele_en_panne)

    with caplog.at_level(logging.ERROR):
        r = client.post("/predict", json=payload_valide)

    # Côté client : rien de sensible
    assert r.status_code == 500
    assert r.json() == {"detail": "Erreur interne"}
    assert "Traceback" not in r.text
    assert "secret" not in r.text
    # Côté serveur : l'erreur est bien journalisée pour le diagnostic
    assert "Erreur interne pendant la prédiction" in caplog.text


def test_probabilite_hors_intervalle_bloquee(client, payload_valide, monkeypatch):
    """Si le modèle renvoie une valeur absurde (1.7), elle n'est JAMAIS transmise."""
    import numpy as np

    monkeypatch.setattr(model_service.model, "predict_proba", lambda X: np.array([[-0.7, 1.7]]))
    r = client.post("/predict", json=payload_valide)
    assert r.status_code == 500
