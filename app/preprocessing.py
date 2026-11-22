"""Module de prétraitement : transforme une demande validée en tableau pour le modèle.

La normalisation (StandardScaler) fait partie du pipeline scikit-learn
sauvegardé avec le modèle : elle est donc appliquée de la même façon
à l'entraînement et en production.
"""

import math

import pandas as pd

from app import config


class PreprocessingError(Exception):
    """Erreur lors de la préparation des variables."""


def construire_features(donnees: dict) -> pd.DataFrame:
    """Retourne un DataFrame d'une ligne, colonnes dans l'ordre config.FEATURES.

    Lève PreprocessingError si une variable manque ou n'est pas un nombre fini.
    """
    manquantes = [f for f in config.FEATURES if f not in donnees]
    if manquantes:
        raise PreprocessingError(f"Variables manquantes : {manquantes}")

    ligne = {}
    for nom in config.FEATURES:
        valeur = donnees[nom]
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
            raise PreprocessingError(f"Variable non numérique : {nom}")
        if not math.isfinite(valeur):
            raise PreprocessingError(f"Variable non finie : {nom}")
        ligne[nom] = float(valeur)

    return pd.DataFrame([ligne], columns=config.FEATURES)
