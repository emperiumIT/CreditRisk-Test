"""Tests UNITAIRES du module de validation (sans passer par l'API).

Exemple corrigé : il montre la méthode à réutiliser dans les autres fichiers.
- @pytest.mark.parametrize : un seul test, exécuté pour chaque valeur de la liste
- pytest.raises             : vérifie qu'une erreur est bien levée
"""

import pytest
from pydantic import ValidationError

from app.validation import DemandeCredit


# EX01 - valeurs limites de l'âge (CT01-1..5)
@pytest.mark.parametrize("age", [18, 35, 75])
def test_age_valide_accepte(payload_valide, age):
    payload_valide["age"] = age
    demande = DemandeCredit(**payload_valide)
    assert demande.age == age


@pytest.mark.parametrize("age", [17, 76, -5, 0])
def test_age_hors_limites_rejete(payload_valide, age):
    payload_valide["age"] = age
    with pytest.raises(ValidationError):
        DemandeCredit(**payload_valide)


# EX02 - revenu (CT02)
@pytest.mark.parametrize("revenu", ["abc", "250000", 0, -1000, 100_000_001, None])
def test_revenu_invalide_rejete(payload_valide, revenu):
    payload_valide["revenu_mensuel"] = revenu
    with pytest.raises(ValidationError):
        DemandeCredit(**payload_valide)


# EX03 - durée (CT03)
@pytest.mark.parametrize("duree", [0, 121, 12.5, "12", -6])
def test_duree_invalide_rejetee(payload_valide, duree):
    payload_valide["duree_mois"] = duree
    with pytest.raises(ValidationError):
        DemandeCredit(**payload_valide)


# EX10 - champ supplémentaire (CT10)
def test_champ_supplementaire_rejete(payload_valide):
    payload_valide["nom"] = "Paul"
    with pytest.raises(ValidationError):
        DemandeCredit(**payload_valide)
