# CreditRisk-Test

Assurance qualité et tests d'un système de prédiction du risque de crédit
(validation → prétraitement → modèle → API FastAPI).

> Les données sont **synthétiques** (`scripts/generate_data.py`). Elles servent à tester
> le système et ne démontrent pas la performance réelle d'un modèle de crédit.

## Structure

```
app/            code de l'application (validation, prétraitement, prédiction, API)
scripts/        génération des données et entraînement du modèle
data/           jeu de données synthétique
models/         modèles versionnés (v1 = référence, v2 = nouvelle version à comparer)
tests/          tests Pytest
reports/        rapports produits (couverture, non-régression, performance)
.github/        workflow GitHub Actions
```

## Installation (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Commandes utiles

| Action | Commande |
|---|---|
| Lancer l'API | `uvicorn app.main:app --reload` puis ouvrir http://127.0.0.1:8000/docs |
| Lancer tous les tests | `pytest` |
| Un seul fichier de tests | `pytest tests/test_api.py` |
| Couverture | `pytest --cov=app --cov-report=term-missing` |
| Qualité du code | `ruff check .` |
| Régénérer les données | `python scripts/generate_data.py` |
| Réentraîner la référence | `python scripts/train.py` |
| Créer la version 2 | `python scripts/train.py --version 2.0.0 --nom model_v2 --C 0.001` |
| Rapport de non-régression v1 vs v2 | `python scripts/compare_models.py` |
| Porte de non-régression sur v2 (PowerShell) | `$env:MODEL_CANDIDAT="model_v2"; pytest tests/test_regression.py` |
| Mesures de performance | lancer l'API, puis `python scripts/perf.py` |
| Test de charge Locust | `pip install -r requirements-perf.txt` puis `locust -f locustfile.py --host http://127.0.0.1:8000` |

## Contrat de l'API

| Endpoint | Rôle |
|---|---|
| `GET /health` | état du service et du modèle |
| `POST /predict` | probabilité de défaut + niveau de risque |
| `GET /model-info` | métadonnées du modèle |

Codes : 200 succès · 422 données invalides · 503 modèle indisponible · 500 erreur interne (message générique).
