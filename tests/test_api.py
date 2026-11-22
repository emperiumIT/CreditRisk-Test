"""Tests fonctionnels de l'API (Partie II).

Chaque test vérifie : le code HTTP, le contenu de la réponse, le comportement.
Les identifiants CTxx renvoient à la matrice de traçabilité du rapport.
"""

import pytest

from app import config

CHAMPS = ["age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"]


# --- Cas nominal (CT04-1, CT11-1) -------------------------------------------
def test_requete_valide(client, payload_valide):
    r = client.post("/predict", json=payload_valide)

    assert r.status_code == 200
    corps = r.json()
    assert set(corps) == {"probabilite_defaut", "niveau_risque", "version_modele", "alertes"}
    assert 0 <= corps["probabilite_defaut"] <= 1
    assert corps["niveau_risque"] in {"faible", "moyen", "eleve"}
    assert corps["version_modele"] == "1.0.0"
    assert corps["alertes"] == []  # dossier ordinaire : aucune alerte


# --- Champ obligatoire manquant (EX06 - CT06-1..5) ----------------------------
@pytest.mark.parametrize("champ", CHAMPS)
def test_champ_manquant(client, payload_valide, champ):
    del payload_valide[champ]
    r = client.post("/predict", json=payload_valide)

    assert r.status_code == 422
    assert champ in str(r.json()["detail"])  # le message nomme le champ fautif


def test_corps_vide(client):  # CT06-6
    r = client.post("/predict", json={})
    assert r.status_code == 422


# --- Valeurs invalides (EX01, EX02, EX03, EX08, EX09) -------------------------
@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("age", -5),  # âge négatif
        ("age", 17),  # juste sous la borne
        ("age", 76),  # juste au-dessus
        ("revenu_mensuel", "abc"),  # revenu en texte
        ("revenu_mensuel", "250000"),  # nombre en texte (validation stricte)
        ("revenu_mensuel", 0),
        ("revenu_mensuel", -1000),
        ("montant_credit", 0),  # montant nul
        ("montant_credit", -1),  # montant négatif
        ("duree_mois", 12.5),  # durée non entière
        ("duree_mois", 0),
        ("duree_mois", 121),
        ("anciennete_activite", -1),
        ("anciennete_activite", "x"),
    ],
)
def test_valeur_invalide(client, payload_valide, champ, valeur):
    payload_valide[champ] = valeur
    r = client.post("/predict", json=payload_valide)

    assert r.status_code == 422
    assert champ in str(r.json()["detail"])


# --- Valeurs aux limites, ACCEPTÉES (analyse des valeurs limites) -------------
@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("age", config.AGE_MIN),
        ("age", config.AGE_MAX),
        ("revenu_mensuel", 1),
        ("revenu_mensuel", config.REVENU_MAX),
        ("montant_credit", config.MONTANT_MAX),
        ("duree_mois", config.DUREE_MIN),
        ("duree_mois", config.DUREE_MAX),
        ("anciennete_activite", 0),
    ],
)
def test_valeur_limite_acceptee(client, payload_valide, champ, valeur):
    payload_valide[champ] = valeur
    r = client.post("/predict", json=payload_valide)
    assert r.status_code == 200


# --- Valeurs extrêmement élevées ----------------------------------------------
@pytest.mark.parametrize(
    "champ, valeur",
    [("revenu_mensuel", 1e15), ("montant_credit", 1e15), ("age", 1_000_000)],
)
def test_valeur_extreme_rejetee(client, payload_valide, champ, valeur):
    payload_valide[champ] = valeur
    r = client.post("/predict", json=payload_valide)
    assert r.status_code == 422


# --- Champ supplémentaire non autorisé (EX10 - CT10-1) ------------------------
def test_champ_supplementaire(client, payload_valide):
    payload_valide["nom"] = "Paul"
    r = client.post("/predict", json=payload_valide)

    assert r.status_code == 422
    assert "nom" in str(r.json()["detail"])


# --- Requête mal formée (CT06-7) ----------------------------------------------
def test_json_mal_forme(client):
    r = client.post("/predict", content='{"age": 35,', headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# --- Modèle indisponible (EX05 - CT05-1) --------------------------------------
def test_predict_modele_indisponible(client_sans_modele, payload_valide):
    r = client_sans_modele.post("/predict", json=payload_valide)

    assert r.status_code == 503
    assert r.json() == {"detail": "Modèle indisponible"}


def test_health_modele_indisponible(client_sans_modele):
    r = client_sans_modele.get("/health")

    assert r.status_code == 200  # le service répond, mais signale le problème
    assert r.json() == {"status": "degraded", "model_loaded": False}


# --- /health et /model-info en fonctionnement normal (EX15 - CT15-1) ----------
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    r = client.get("/model-info")

    assert r.status_code == 200
    info = r.json()
    assert info["version"] == "1.0.0"
    assert info["features"] == config.FEATURES  # même ordre que le prétraitement
    assert "date_entrainement" in info
