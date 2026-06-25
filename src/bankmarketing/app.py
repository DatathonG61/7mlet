"""API FastAPI para recomendação de ofertas (E5).

Endpoint principal:
  POST /decide — recebe contexto do cliente, retorna oferta recomendada + log de auditoria

Suporte:
  GET /health — status da API
  GET /log — últimas decisões (para inspeção)

A API é stateless: a política e o log são gerenciados em camadas separadas.
Integração com LLM (explicações) vem na semana 6 (E5.5).
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from bankmarketing.contracts import ClientContext, DecisionResponse, HealthResponse
from bankmarketing.decision_log import clear_log, count_log, log_decision, retrieve_log
from bankmarketing.policies import get_policy
from bankmarketing.synthetic import OFFER_IDS, assign_segments

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BankMarketing — Plataforma de Experimentação Adaptativa",
    description="API de decisão de ofertas financeiras com log de auditoria",
    version="0.1.0",
)

# Política ativa (configurável via query param na demo, seria env var em produção)
ACTIVE_POLICY = "baseline"


@app.get("/health", response_model=HealthResponse)
def health():
    """Status da API e contadores."""
    try:
        log_count = count_log()
        return HealthResponse(status="ok", version="0.1.0", log_entry_count=log_count)
    except Exception as e:
        logger.error("Health check falhou: %s", e)
        return HealthResponse(status="degraded", version="0.1.0", log_entry_count=0)


@app.post("/decide", response_model=DecisionResponse)
def decide(
    client: ClientContext,
    policy: Literal["baseline", "thompson_sampling"] = Query(
        "baseline",
        description="Política a usar (baseline | thompson_sampling)",
    ),
):
    """Recomenda uma oferta para o cliente.

    Request body: ClientContext (age, job, default, segment, propensity_score, etc.)
    Response: DecisionResponse (offer_id, rationale, decision_id para auditoria)

    Args:
        client: contexto do cliente (Pydantic).
        policy: política ativa ("baseline" por enquanto; "thompson_sampling" em E6).

    Returns:
        DecisionResponse com recomendação + decision_id para auditoria.

    Raises:
        HTTPException: se política não existe ou dados inválidos.
    """
    try:
        # Valida política
        if policy not in ["baseline", "thompson_sampling"]:
            raise ValueError(f"Política desconhecida: {policy}")

        # Instancia a política e obtém decisão
        policy_obj = get_policy(policy)
        decision = policy_obj.decide(
            age=client.age,
            segment=client.segment,
            propensity_score=client.propensity_score,
            default=client.default,
        )

        # Valida offer_id (segurança)
        if decision.offer_id not in OFFER_IDS:
            raise ValueError(f"offer_id inválido: {decision.offer_id}")

        # Registra no log de auditoria
        log_entry = log_decision(
            client_id=client.client_id,
            age=client.age,
            job=client.job,
            default=client.default,
            segment=client.segment,
            propensity_score=client.propensity_score,
            policy=policy,
            offer_id=decision.offer_id,
            rationale=decision.rationale,
            confidence=decision.confidence,
            clicked=0,  # preenchido offline
        )

        # Retorna resposta
        return DecisionResponse(
            decision_id=log_entry.decision_id,
            timestamp=log_entry.timestamp,
            offer_id=decision.offer_id,
            offer_name=log_entry.offer_name,
            policy=policy,
            rationale=decision.rationale,
            confidence=decision.confidence,
        )

    except ValueError as e:
        logger.warning("Erro de validação: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Erro ao processar /decide: %s", e)
        raise HTTPException(status_code=500, detail="Erro interno do servidor") from e


@app.get("/log")
def get_log_entries(
    limit: int = Query(10, ge=1, le=1000),
    policy: str | None = Query(None, description="Filtrar por política"),
    offer_id: str | None = Query(None, description="Filtrar por oferta"),
    segment: str | None = Query(None, description="Filtrar por segmento"),
):
    """Recupera últimas decisões do log (com filtros opcionais).

    Uso: inspecionar histórico de decisões, debug, explicações para LLM.
    """
    try:
        entries = retrieve_log(
            limit=limit,
            policy=policy,
            offer_id=offer_id,
            segment=segment,
        )
        return {
            "count": len(entries),
            "entries": [e.model_dump() for e in entries],
        }
    except Exception as e:
        logger.error("Erro ao recuperar log: %s", e)
        raise HTTPException(status_code=500, detail="Erro ao recuperar log") from e


@app.post("/admin/clear-log")
def admin_clear_log():
    """Limpa o log (apenas para testes/demo)."""
    try:
        clear_log()
        return {"message": "Log limpo com sucesso"}
    except Exception as e:
        logger.error("Erro ao limpar log: %s", e)
        raise HTTPException(status_code=500, detail="Erro ao limpar log") from e


@app.get("/")
def root():
    """Root endpoint — informações da API."""
    return {
        "project": "BankMarketing — Experimentação Adaptativa de Ofertas",
        "version": "0.1.0",
        "endpoints": {
            "GET /health": "Status da API",
            "POST /decide": "Recomenda uma oferta (ClientContext → DecisionResponse)",
            "GET /log": "Recupera histórico de decisões",
            "POST /admin/clear-log": "Limpa o log (debug/teste)",
        },
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Roda a API em localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
