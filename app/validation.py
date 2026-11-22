"""Module de validation : contrat d'entrée et de sortie de l'API.

Pydantic vérifie automatiquement le type et les bornes de chaque champ.
- strict=True        : "12" (texte) n'est PAS converti en 12 -> rejeté
- extra="forbid"     : un champ non prévu est rejeté
- allow_inf_nan=False: NaN et Infinity sont rejetés
Si la validation échoue, FastAPI répond 422 automatiquement.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app import config


class DemandeCredit(BaseModel):
    """Données d'un demandeur de crédit."""

    model_config = ConfigDict(extra="forbid", strict=True)

    age: int = Field(ge=config.AGE_MIN, le=config.AGE_MAX, examples=[35])
    revenu_mensuel: float = Field(
        gt=0, le=config.REVENU_MAX, allow_inf_nan=False, examples=[250000]
    )
    montant_credit: float = Field(
        gt=0, le=config.MONTANT_MAX, allow_inf_nan=False, examples=[1000000]
    )
    duree_mois: int = Field(ge=config.DUREE_MIN, le=config.DUREE_MAX, examples=[12])
    anciennete_activite: float = Field(
        ge=config.ANCIENNETE_MIN,
        le=config.ANCIENNETE_MAX,
        allow_inf_nan=False,
        examples=[4],
    )


class ResultatPrediction(BaseModel):
    """Réponse de POST /predict (contrat de sortie)."""

    probabilite_defaut: float = Field(ge=0.0, le=1.0)
    niveau_risque: Literal["faible", "moyen", "eleve"]
    version_modele: str
    alertes: list[str] = Field(
        default_factory=list,
        description="Vide si le dossier est ordinaire ; sinon, revue manuelle conseillée",
    )
