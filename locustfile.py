"""Test de charge Locust (optionnel, Partie VI).

pip install -r requirements-perf.txt
uvicorn app.main:app --port 8000
locust -f locustfile.py --host http://127.0.0.1:8000 --headless -u 20 -r 5 -t 30s
"""

from locust import HttpUser, between, task


class AgentDeCredit(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predire(self):
        self.client.post(
            "/predict",
            json={
                "age": 35,
                "revenu_mensuel": 250000,
                "montant_credit": 1000000,
                "duree_mois": 12,
                "anciennete_activite": 4,
            },
        )

    @task(1)
    def sante(self):
        self.client.get("/health")
