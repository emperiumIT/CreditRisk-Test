"""Mesures de performance (Partie VI) : latence d'inférence, latence API, P95, débit, charge.

Usage :
  1) uvicorn app.main:app --port 8000      (dans un autre terminal)
  2) python scripts/perf.py
Résultats : reports/performance.json et reports/performance.md
"""

import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

from app import config  # noqa: E402
from app.prediction import ModelService  # noqa: E402
from app.preprocessing import construire_features  # noqa: E402

URL = "http://127.0.0.1:8000/predict"
DOSSIER = {
    "age": 35,
    "revenu_mensuel": 250000,
    "montant_credit": 1000000,
    "duree_mois": 12,
    "anciennete_activite": 4,
}
OBJECTIF_P95_MS = 200


def resume(latences_ms):
    a = np.array(latences_ms)
    return {
        "n": len(a),
        "moyenne_ms": round(float(a.mean()), 2),
        "mediane_ms": round(float(np.percentile(a, 50)), 2),
        "p95_ms": round(float(np.percentile(a, 95)), 2),
        "max_ms": round(float(a.max()), 2),
    }


def latence_inference(n=1000):
    service = ModelService()
    service.load(config.MODEL_PATH)
    X = construire_features(DOSSIER)
    for _ in range(20):  # échauffement
        service.predict_proba(X)
    mesures = []
    for _ in range(n):
        t = time.perf_counter()
        service.predict_proba(X)
        mesures.append((time.perf_counter() - t) * 1000)
    return resume(mesures)


def charge(client, utilisateurs, n_requetes):
    def une_requete(_):
        t = time.perf_counter()
        r = client.post(URL, json=DOSSIER)
        return (time.perf_counter() - t) * 1000, r.status_code

    debut = time.perf_counter()
    with ThreadPoolExecutor(max_workers=utilisateurs) as pool:
        resultats = list(pool.map(une_requete, range(n_requetes)))
    duree = time.perf_counter() - debut
    lat = [ms for ms, _ in resultats]
    erreurs = sum(1 for _, code in resultats if code != 200)
    return {
        "utilisateurs": utilisateurs,
        **resume(lat),
        "debit_req_s": round(n_requetes / duree, 1),
        "erreurs": erreurs,
    }


def main():
    res = {
        "configuration": {
            "os": f"{platform.system()} {platform.release()}",
            "processeur": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "serveur": "uvicorn, 1 worker, local (127.0.0.1)",
        },
        "objectif": f"P95 de /predict < {OBJECTIF_P95_MS} ms",
        "inference_modele": latence_inference(),
    }
    with httpx.Client(timeout=10) as client:
        for _ in range(20):
            client.post(URL, json=DOSSIER)
        res["api_sequentiel"] = charge(client, 1, 500)
        res["charge_progressive"] = [charge(client, u, 500) for u in (1, 5, 10, 20, 50)]

    p95 = res["api_sequentiel"]["p95_ms"]
    res["objectif_respecte"] = p95 < OBJECTIF_P95_MS

    (RACINE / "reports").mkdir(exist_ok=True)
    (RACINE / "reports" / "performance.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    verdict = "RESPECTÉ" if res["objectif_respecte"] else "NON RESPECTÉ"
    print(f"\nObjectif P95 < {OBJECTIF_P95_MS} ms : {verdict}")


if __name__ == "__main__":
    main()
