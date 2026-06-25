"""Políticas de decisão (E3, E5).

Define as estratégias de recomendação de ofertas:
- BaselinePolicy: regra fixa (if/else simples por segmento)
- ThompsonSamplingPolicy: exploração bayesiana (implementado em E3 por Adryen)
- UCBPolicy: Upper Confidence Bound (futuro)

Cada política é stateless na API; a E7 (MLOps) gerencia versões e retreino offline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from bankmarketing.contracts import VALID_OFFERS, VALID_SEGMENTS


@dataclass(frozen=True)
class PolicyDecision:
    """Resultado de uma decisão de política."""

    offer_id: VALID_OFFERS
    rationale: str
    confidence: float = 0.5  # default para baseline (determinístico, confiança alta seria 0.9+)


class BasePolicy(ABC):
    """Interface abstrata para políticas de recomendação."""

    @abstractmethod
    def decide(
        self,
        age: int,
        segment: VALID_SEGMENTS,
        propensity_score: float,
        default: str,
    ) -> PolicyDecision:
        """Recomenda uma oferta baseada no contexto do cliente.

        Args:
            age: idade do cliente.
            segment: segmento sintético.
            propensity_score: score de propensão (0–1).
            default: status de inadimplência ("yes" ou "no").

        Returns:
            PolicyDecision com offer_id, rationale e confidence.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome legível da política."""
        pass


class BaselinePolicy(BasePolicy):
    """Baseline: regra if/else simples baseada em segmento e situação de inadimplência.

    Hipóteses de negócio (do PLANO.md):
    - Negativado (default=yes) → renegociacao (nunca crédito)
    - Maduro + alta renda (age≥50, job↑) → investimento
    - Jovem + baixa renda (age<35, job↓) → cartao_credito
    - Geral → sem_oferta (controle)
    """

    def decide(
        self,
        age: int,
        segment: VALID_SEGMENTS,
        propensity_score: float,
        default: str,
    ) -> PolicyDecision:
        """Aplica regras de negócio predefinidas."""

        # Prioridade 1: Negativado nunca recebe crédito
        if default == "yes":
            return PolicyDecision(
                offer_id="renegociacao",
                rationale="Cliente com indicativo de inadimplência (default=yes). Oferta apropriada: plano de renegociação.",
                confidence=0.95,
            )

        # Prioridade 2: Segmento maduro + alta renda
        if segment == "maduro_alta_renda":
            return PolicyDecision(
                offer_id="investimento",
                rationale=f"Segmento maduro de alta renda (age={age}). Hipótese: préferência por investimento.",
                confidence=0.90,
            )

        # Prioridade 3: Segmento jovem + baixa renda
        if segment == "jovem_baixa_renda":
            return PolicyDecision(
                offer_id="cartao_credito",
                rationale=f"Segmento jovem de baixa renda (age={age}). Hipótese: preferência por cartão de crédito.",
                confidence=0.85,
            )

        # Fallback: Controle (não ofertar)
        return PolicyDecision(
            offer_id="sem_oferta",
            rationale=f"Segmento geral (propensity={propensity_score:.2f}). Braço de controle: nenhuma oferta.",
            confidence=0.50,
        )

    @property
    def name(self) -> str:
        return "baseline"


class ThompsonSamplingPolicy(BasePolicy):
    """Thompson Sampling: exploração bayesiana.

    Implementado em E3 por Adryen. Placeholder para integração posterior.
    """

    def __init__(self, model_path: str | None = None):
        """Carrega modelo Thompson Sampling (stub por enquanto)."""
        self.model_path = model_path
        self._model = None

    def decide(
        self,
        age: int,
        segment: VALID_SEGMENTS,
        propensity_score: float,
        default: str,
    ) -> PolicyDecision:
        """Thompson Sampling (stub — implementado em E3)."""
        raise NotImplementedError(
            "Thompson Sampling será implementado na E3 (Adryen). "
            "Para agora, use BaselinePolicy."
        )

    @property
    def name(self) -> str:
        return "thompson_sampling"


# Factory para instanciar políticas
POLICIES: dict[str, type[BasePolicy]] = {
    "baseline": BaselinePolicy,
    "thompson_sampling": ThompsonSamplingPolicy,
}


def get_policy(name: str) -> BasePolicy:
    """Retorna uma instância da política pelo nome.

    Args:
        name: nome da política ("baseline", "thompson_sampling", etc.).

    Returns:
        Instância da política.

    Raises:
        ValueError: se política não existe.
    """
    if name not in POLICIES:
        raise ValueError(
            f"Política '{name}' desconhecida. Disponíveis: {list(POLICIES.keys())}"
        )
    return POLICIES[name]()
