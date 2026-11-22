"""Fixtures partagées par tous les tests.

Une fixture est une "préparation" réutilisable : Pytest l'injecte
automatiquement dans tout test qui la demande en paramètre.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.prediction import ModelService, get_model_service


@pytest.fixture
def payload_valide() -> dict:
    """Une demande valide de référence. Chaque test en modifie UN SEUL champ."""
    return {
        "age": 35,
        "revenu_mensuel": 250000,
        "montant_credit": 1000000,
        "duree_mois": 12,
        "anciennete_activite": 4,
    }


@pytest.fixture
def client():
    """Client HTTP de test. Le bloc `with` déclenche le chargement du modèle."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_sans_modele():
    """Client dont le service de modèle est VIDE (simule un modèle non chargé).

    dependency_overrides remplace la dépendance get_model_service
    par un faux service, uniquement pendant ce test.
    """
    service_vide = ModelService()  # model = None
    app.dependency_overrides[get_model_service] = lambda: service_vide
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
