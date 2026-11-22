"""API de prédiction du risque de crédit (FastAPI).

Lancer en local :  uvicorn app.main:app --reload
Documentation   :  http://127.0.0.1:8000/docs
"""

import logging
import math
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app import config
from app.prediction import (
    ModelNotLoadedError,
    ModelService,
    detecter_alertes,
    get_model_service,
    model_service,
    niveau_risque,
)
from app.preprocessing import construire_features
from app.validation import DemandeCredit, ResultatPrediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("creditrisk.api")

# Dépendance injectée : les tests peuvent la remplacer (app.dependency_overrides)
ServiceDep = Annotated[ModelService, Depends(get_model_service)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Au démarrage : tente de charger le modèle. En cas d'échec, l'API démarre
    quand même en mode dégradé (/health le signale, /predict répond 503)."""
    try:
        model_service.load(config.MODEL_PATH)
    except Exception:
        logger.exception("Impossible de charger le modèle : %s", config.MODEL_PATH)
    yield


app = FastAPI(
    title="CreditRisk API",
    version="1.0.0",
    description="Estimation de la probabilité de défaut d'un demandeur de crédit.",
    lifespan=lifespan,
)


def _assainir(valeur):
    """Remplace NaN / Infinity par du texte pour que la réponse reste du JSON valide."""
    if isinstance(valeur, float) and not math.isfinite(valeur):
        return str(valeur)
    if isinstance(valeur, dict):
        return {cle: _assainir(v) for cle, v in valeur.items()}
    if isinstance(valeur, (list, tuple)):
        return [_assainir(v) for v in valeur]
    return valeur


@app.exception_handler(RequestValidationError)
async def erreur_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Correction ANO-01 : le handler par défaut plantait (500) quand l'entrée
    rejetée contenait NaN ou Infinity, car ces valeurs ne sont pas du JSON valide."""
    return JSONResponse(
        status_code=422, content={"detail": _assainir(jsonable_encoder(exc.errors()))}
    )


@app.get("/health")
def health(service: ServiceDep) -> dict:
    """Disponibilité du service et du modèle."""
    return {
        "status": "ok" if service.is_loaded else "degraded",
        "model_loaded": service.is_loaded,
    }


@app.get("/model-info")
def model_info(service: ServiceDep) -> dict:
    """Métadonnées de la version du modèle."""
    if not service.is_loaded:
        raise HTTPException(status_code=503, detail="Modèle indisponible")
    return service.metadata


@app.post("/predict", response_model=ResultatPrediction)
def predict(demande: DemandeCredit, service: ServiceDep) -> ResultatPrediction:
    """Retourne la probabilité de défaut et le niveau de risque.

    Arrivé ici, `demande` a déjà été validée par Pydantic (sinon : 422).
    """
    try:
        donnees = demande.model_dump()
        X = construire_features(donnees)
        probabilite = service.predict_proba(X)
        alertes = detecter_alertes(donnees, service.metadata)
    except ModelNotLoadedError:
        raise HTTPException(status_code=503, detail="Modèle indisponible") from None
    except Exception:
        # La trace complète va dans les logs serveur, jamais au client (EX13)
        logger.exception("Erreur interne pendant la prédiction")
        raise HTTPException(status_code=500, detail="Erreur interne") from None

    return ResultatPrediction(
        probabilite_defaut=probabilite,
        niveau_risque=niveau_risque(probabilite),
        version_modele=service.metadata.get("version", "inconnue"),
        alertes=alertes,
    )
