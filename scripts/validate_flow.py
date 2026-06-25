#!/usr/bin/env python
"""Validação do fluxo E5 com saída passo a passo.

Este script não é um teste pytest: ele demonstra cada etapa do fluxo em sequência.
Use:
    python scripts/validate_flow.py
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi.testclient import TestClient

from bankmarketing.app import app
from bankmarketing.contracts import ClientContext
from bankmarketing.decision_log import clear_log, count_log, log_decision, retrieve_log
from bankmarketing.policies import BaselinePolicy


def make_client_context(client_id, age, job, default, loan, housing, segment, propensity_score):
    return ClientContext(
        client_id=client_id,
        age=age,
        job=job,
        default=default,
        loan=loan,
        housing=housing,
        segment=segment,
        propensity_score=propensity_score,
    )


def main():
    clear_log()

    print("\n=== STEP 1: Contract validation ===")
    client = make_client_context(
        client_id=1,
        age=28,
        job="student",
        default="no",
        loan="no",
        housing="no",
        segment="jovem_baixa_renda",
        propensity_score=0.45,
    )
    print("ClientContext:", client.model_dump())

    print("\n=== STEP 2: Policy decision ===")
    policy = BaselinePolicy()
    decision = policy.decide(
        age=client.age,
        segment=client.segment,
        propensity_score=client.propensity_score,
        default=client.default,
    )
    print("Policy:", policy.name)
    print(
        "Decision:",
        {
            "offer_id": decision.offer_id,
            "rationale": decision.rationale,
            "confidence": decision.confidence,
        },
    )

    print("\n=== STEP 3: Audit log write ===")
    entry = log_decision(
        client_id=client.client_id,
        age=client.age,
        job=client.job,
        default=client.default,
        segment=client.segment,
        propensity_score=client.propensity_score,
        policy=policy.name,
        offer_id=decision.offer_id,
        rationale=decision.rationale,
        confidence=decision.confidence,
    )
    print("Logged decision_id:", entry.decision_id)
    print("Log count after write:", count_log())

    print("\n=== STEP 4: API request ===")
    api_client = TestClient(app)
    response = api_client.post(
        "/decide",
        json=client.model_dump(),
        params={"policy": "baseline"},
    )
    print("API status_code:", response.status_code)
    print("API response:", response.json())

    print("\n=== STEP 5: Retrieve audit history ===")
    history = retrieve_log(limit=5)
    print(f"Retrieved {len(history)} entries")
    for item in history:
        print(item.model_dump())

    clear_log()
    print("\nValidação concluída.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
