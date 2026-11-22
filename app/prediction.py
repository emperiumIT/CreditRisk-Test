"""Module de prédiction : chargement du modèle et calcul de la probabilité de défaut."""

import json
import logging
import math
from pathlib import Path

import joblib
import pandas as pd

from app import config

logger = logging.getLogger(__name__)


class ModelNotLoadedError(Exception):
    """Le modèle n'est pas disponible."""


class PredictionError(Exception):
    """Erreur pendant le calcul de la prédiction."""


def niveau_risque(probabilite: float) -> str:
    """Traduit une probabilité en niveau de risque lisible par l'agent de crédit."""
    if probabilite >= config.SEUIL_ELEVE:
        return "eleve"
    if probabilite >= config.SEUIL_MOYEN:
        return "moyen"
    return "faible"


def detecter_alertes(donnees: dict, metadata: dict) -> list[str]:
    """Signale les entrées VALIDES mais INHABITUELLES (correction ANO-02).

    - HORS_DOMAINE:<variable> : valeur hors de la plage vue à l'entraînement,
      la probabilité est alors une extrapolation non fiable ;
    - TAUX_EFFORT_ELEVE : mensualité / revenu au-delà du seuil métier.
    """
    alertes = []
    domaine = metadata.get("domaine_entrainement", {})
    for variable, (mini, maxi) in domaine.items():
        if variable in donnees and not mini <= donnees[variable] <= maxi:
            alertes.append(f"HORS_DOMAINE:{variable}")

    taux_effort = (donnees["montant_credit"] / donnees["duree_mois"]) / donnees["revenu_mensuel"]
    if taux_effort > config.SEUIL_TAUX_EFFORT:
        alertes.append("TAUX_EFFORT_ELEVE")
    return alertes


class ModelService:
    """Encapsule le modèle et ses métadonnées."""

    def __init__(self) -> None:
        self.model = None
        self.metadata: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self, chemin: Path) -> None:
        """Charge le modèle (.joblib) et ses métadonnées (.json à côté)."""
        chemin = Path(chemin)
        self.model = joblib.load(chemin)
        meta = chemin.with_suffix(".json")
        self.metadata = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        logger.info("Modèle chargé : %s", chemin.name)

    def predict_proba(self, X: pd.DataFrame) -> float:
        """Retourne la probabilité de défaut (classe 1) pour une ligne."""
        if not self.is_loaded:
            raise ModelNotLoadedError("Modèle non chargé")

        attendu = getattr(self.model, "n_features_in_", X.shape[1])
        if X.shape[1] != attendu:
            raise PredictionError(
                f"Dimensions incompatibles : {X.shape[1]} reçues, {attendu} attendues"
            )

        try:
            probabilite = float(self.model.predict_proba(X)[0, 1])
        except Exception as exc:  # toute erreur du modèle devient une PredictionError
            raise PredictionError("Échec du calcul de la prédiction") from exc

        if not math.isfinite(probabilite) or not 0.0 <= probabilite <= 1.0:
            raise PredictionError(f"Probabilité invalide : {probabilite}")
        return probabilite


# Instance unique utilisée par l'API (remplaçable dans les tests)
model_service = ModelService()


def get_model_service() -> ModelService:
    """Dépendance FastAPI : permet aux tests d'injecter un faux service."""
    return model_service
