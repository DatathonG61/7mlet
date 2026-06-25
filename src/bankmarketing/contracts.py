"""Contratos e esquemas compartilhados (E5).

Define os modelos Pydantic que circulam entre:
- Cliente HTTP → API /decide
- API → Log de Auditoria
- Log → Exportação (Parquet para replay E4/E7)
- Log → Assistente LLM

Vocabulário compartilhado:
- `offer_id`: sempre um dos 4 braços (sem_oferta, cartao_credito, investimento, renegociacao)
- `segment`: sempre um dos 4 segmentos sintéticos (jovem_baixa_renda, maduro_alta_renda, negativado, geral)
- `policy`: nome legível da política que decidiu (baseline, thompson_sampling, ucb, etc.)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

VALID_OFFERS = Literal["sem_oferta", "cartao_credito", "investimento", "renegociacao"]
VALID_SEGMENTS = Literal["jovem_baixa_renda", "maduro_alta_renda", "negativado", "geral"]
VALID_POLICIES = Literal["baseline", "thompson_sampling", "ucb"]


class ClientContext(BaseModel):
    """Contexto do cliente para recomendação de oferta (input à API).

    Captura o mínimo necessário da modeling_table para a política decidir.
    """

    client_id: int = Field(..., description="Índice/ID do cliente")
    age: int = Field(..., ge=18, le=120, description="Idade do cliente")
    job: str = Field(..., description="Profissão (ex: student, management, etc.)")
    default: Literal["yes", "no"] = Field(..., description="Status de inadimplência")
    loan: Literal["yes", "no"] = Field(..., description="Tem empréstimo ativo?")
    housing: Literal["yes", "no"] = Field(..., description="Tem financiamento habitacional?")
    segment: VALID_SEGMENTS = Field(..., description="Segmento sintético (derivado de age/job/default)")
    propensity_score: float = Field(..., ge=0.0, le=1.0, description="Score de propensão da E1")


class OfferInfo(BaseModel):
    """Informação sobre uma oferta (parte do catálogo)."""

    offer_id: VALID_OFFERS
    offer_name: str = Field(..., description="Nome legível da oferta")
    public_target: str = Field(..., description="Público alvo esperado (ex: jovem_baixa_renda)")


class DecisionResponse(BaseModel):
    """Resposta da API /decide com a recomendação e rastreabilidade."""

    decision_id: str = Field(..., description="UUID único da decisão (para auditoria)")
    timestamp: str = Field(..., description="ISO 8601 timestamp UTC")
    offer_id: VALID_OFFERS = Field(..., description="Oferta recomendada")
    offer_name: str = Field(..., description="Nome legível da oferta")
    policy: VALID_POLICIES = Field(..., description="Política que gerou a recomendação")
    rationale: str = Field(..., description="Explicação em texto livre do motivo da recomendação")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiança da recomendação (0–1)")


class DecisionLogEntry(BaseModel):
    """Entrada no log de auditoria (para exportação e replay)."""

    decision_id: str
    timestamp: str
    client_id: int
    age: int
    job: str
    default: Literal["yes", "no"]
    segment: VALID_SEGMENTS
    propensity_score: float
    policy: VALID_POLICIES
    offer_id: VALID_OFFERS
    offer_name: str
    rationale: str
    confidence: float
    clicked: int = Field(0, description="1 se cliente clicou, 0 caso contrário (preenchido offline)")
    reward: int = Field(0, description="1 se conversão, 0 caso contrário (delayed, preenchido offline)")
    delay_days: int = Field(0, description="Dias até conversão materializar (preenchido offline)")


class HealthResponse(BaseModel):
    """Resposta do endpoint GET /health."""

    status: Literal["ok", "degraded"]
    version: str
    log_entry_count: int
