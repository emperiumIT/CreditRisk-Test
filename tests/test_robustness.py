"""Tests de robustesse (Partie IV).

Deux familles d'entrées à distinguer :
- INVALIDES               -> doivent être REJETÉES (422), jamais planter ;
- INHABITUELLES MAIS VALIDES -> acceptées (200), mais SIGNALÉES par une alerte.

Anomalies du registre couvertes ici :
- ANO-01 : NaN / Infinity faisaient planter le handler de validation (500 au lieu de 422)
- ANO-02 : profil hors domaine d'entraînement classé "faible" sans aucun signalement
"""

import math

import pytest

from app import config

CHAMPS = ["age", "revenu_mensuel", "montant_credit", "duree_mois", "anciennete_activite"]
HEADERS_JSON = {"Content-Type": "application/json"}


def corps_brut(payload: dict, champ: str, valeur_brute: str) -> str:
    """Construit un JSON texte en injectant une valeur brute (NaN, Infinity...)."""
    morceaux = [f'"{c}": {valeur_brute if c == champ else payload[c]}' for c in CHAMPS]
    return "{" + ", ".join(morceaux) + "}"


# ============================================================================
# Contrat de sortie
# ============================================================================


def test_probabilite_finie_et_bornee(client, payload_valide):
    response = client.post("/predict", json=payload_valide)
    assert response.status_code == 200
    p = response.json()["probabilite_defaut"]
    assert math.isfinite(p)
    assert 0.0 <= p <= 1.0


# ============================================================================
# 1. Valeurs manquantes (null explicite)
# ============================================================================


@pytest.mark.parametrize("champ", CHAMPS)
def test_valeur_null_rejetee(client, payload_valide, champ):
    payload_valide[champ] = None
    r = client.post("/predict", json=payload_valide)
    assert r.status_code == 422


# ============================================================================
# 2. Valeurs non numériques
# ============================================================================


@pytest.mark.parametrize("valeur", ["abc", "", [35], {"v": 35}, True])
@pytest.mark.parametrize("champ", CHAMPS)
def test_valeur_non_numerique_rejetee(client, payload_valide, champ, valeur):
    payload_valide[champ] = valeur
    r = client.post("/predict", json=payload_valide)
    assert r.status_code == 422


# ============================================================================
# 3. Valeurs infinies ou non finies (ANO-01)
# ============================================================================


def test_revenu_nan(client):
    corps = (
        '{"age": 35, "revenu_mensuel": NaN, "montant_credit": 1000000, '
        '"duree_mois": 12, "anciennete_activite": 4}'
    )
    r = client.post("/predict", content=corps, headers=HEADERS_JSON)
    assert r.status_code == 422


@pytest.mark.parametrize("valeur_brute", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("champ", CHAMPS)
def test_valeur_non_finie_rejetee_sans_crash(client, payload_valide, champ, valeur_brute):
    r = client.post(
        "/predict", content=corps_brut(payload_valide, champ, valeur_brute), headers=HEADERS_JSON
    )
    assert r.status_code == 422  # et surtout pas 500
    assert champ in str(r.json()["detail"])


# ============================================================================
# 4. Valeurs aux limites (juste dedans = 200, juste dehors = 422)
# ============================================================================

LIMITES = [
    ("age", config.AGE_MIN, config.AGE_MIN - 1),
    ("age", config.AGE_MAX, config.AGE_MAX + 1),
    ("revenu_mensuel", config.REVENU_MAX, config.REVENU_MAX + 1),
    ("montant_credit", config.MONTANT_MAX, config.MONTANT_MAX + 1),
    ("duree_mois", config.DUREE_MIN, config.DUREE_MIN - 1),
    ("duree_mois", config.DUREE_MAX, config.DUREE_MAX + 1),
    ("anciennete_activite", config.ANCIENNETE_MIN, config.ANCIENNETE_MIN - 0.1),
    ("anciennete_activite", config.ANCIENNETE_MAX, config.ANCIENNETE_MAX + 0.1),
]


@pytest.mark.parametrize("champ, dedans, dehors", LIMITES)
def test_limites(client, payload_valide, champ, dedans, dehors):
    payload_valide[champ] = dedans
    assert client.post("/predict", json=payload_valide).status_code == 200
    payload_valide[champ] = dehors
    assert client.post("/predict", json=payload_valide).status_code == 422


def test_revenu_plus_petit_positif_accepte(client, payload_valide):
    payload_valide["revenu_mensuel"] = 0.01
    assert client.post("/predict", json=payload_valide).status_code == 200


# ============================================================================
# 5. Entrées valides mais très éloignées des données d'entraînement (ANO-02)
# ============================================================================


def test_dossier_ordinaire_sans_alerte(client, payload_valide):
    r = client.post("/predict", json=payload_valide)
    assert r.json()["alertes"] == []


def test_profil_extreme_mais_valide(client, payload_valide):
    """500 M à rembourser en 1 mois avec 100 M de revenu : mensualité = 5x le revenu.

    Avant correction : 200, p ~ 1e-35, niveau "faible", AUCUN signalement.
    Attendu : accepté (valeur autorisée) MAIS signalé pour revue manuelle.
    """
    payload_valide.update(revenu_mensuel=100_000_000, montant_credit=500_000_000, duree_mois=1)
    r = client.post("/predict", json=payload_valide)

    assert r.status_code == 200
    alertes = r.json()["alertes"]
    assert "HORS_DOMAINE:revenu_mensuel" in alertes
    assert "HORS_DOMAINE:montant_credit" in alertes
    assert "HORS_DOMAINE:duree_mois" in alertes
    assert "TAUX_EFFORT_ELEVE" in alertes


@pytest.mark.parametrize(
    "champ, valeur",
    [
        ("revenu_mensuel", 50_000_000),  # > max entraînement (1,745 M)
        ("montant_credit", 1_000),  # < min entraînement (50 000)
        ("duree_mois", 120),  # > max entraînement (60)
        ("anciennete_activite", 55),  # > max entraînement (40)
    ],
)
def test_hors_domaine_signale(client, payload_valide, champ, valeur):
    payload_valide[champ] = valeur
    r = client.post("/predict", json=payload_valide)
    assert r.status_code == 200
    assert f"HORS_DOMAINE:{champ}" in r.json()["alertes"]


def test_taux_effort_eleve_signale_meme_dans_le_domaine(client, payload_valide):
    # 3 000 000 sur 6 mois = 500 000 / mois pour 250 000 de revenu -> 200 %
    payload_valide.update(montant_credit=3_000_000, duree_mois=6)
    r = client.post("/predict", json=payload_valide)
    assert r.json()["alertes"] == ["TAUX_EFFORT_ELEVE"]


# ============================================================================
# 6. Schéma incorrect
# ============================================================================


@pytest.mark.parametrize(
    "corps",
    [
        "[]",  # tableau au lieu d'objet
        '"texte"',  # chaîne seule
        "35",  # nombre seul
        "null",
        '{"demande": {"age": 35}}',  # structure imbriquée
        "",  # corps vide
    ],
)
def test_schema_incorrect_rejete(client, corps):
    r = client.post("/predict", content=corps, headers=HEADERS_JSON)
    assert r.status_code == 422


def test_mauvais_type_de_contenu(client):
    r = client.post("/predict", content="age=35", headers={"Content-Type": "text/plain"})
    assert r.status_code == 422


def test_methode_non_autorisee(client):
    assert client.get("/predict").status_code == 405
