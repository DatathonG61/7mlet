"""Camada de enriquecimento sintético (E2).

Sobre a base factual de propensão (`modeling_table`), este módulo fabrica a
camada de experimentação adaptativa que alimenta o bandit (E3):

- ``offer_catalog``     — catálogo estático dos braços (ofertas).
- ``offer_events``      — um evento por cliente: braço apresentado + clique.
- ``delayed_rewards``   — recompensa (conversão) que chega com atraso.

Todo o processo gerador é controlado por *seeds* (``numpy.random.default_rng``)
para garantir reprodutibilidade. As hipóteses de negócio e a calibração do
gerador estão documentadas em ``reports/data-generation.md``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SYNTH_DIR = ROOT / "data" / "synthetic_enrichment"
OFFER_CATALOG_PATH = SYNTH_DIR / "offer_catalog.csv"


@dataclass(frozen=True)
class Offer:
    """Um braço do bandit (oferta apresentável ao cliente)."""

    offer_id: str
    nome: str
    publico_alvo: str
    descricao: str


# Catálogo canônico de ofertas. O vocabulário de `offer_id` é compartilhado com
# policies.py (E3), contracts.py (E5) e o golden set (E4) — ver CLAUDE.md.
OFFER_CATALOG: tuple[Offer, ...] = (
    Offer(
        offer_id="sem_oferta",
        nome="Controle — não ofertar",
        publico_alvo="todos",
        descricao="Braço de controle: nenhuma oferta é apresentada ao cliente.",
    ),
    Offer(
        offer_id="cartao_credito",
        nome="Oferta A — Cartão de Crédito",
        publico_alvo="jovem_baixa_renda",
        descricao="Cartão de crédito com limite inicial; hipótese: atrai perfil jovem.",
    ),
    Offer(
        offer_id="investimento",
        nome="Oferta B — Investimento",
        publico_alvo="maduro_alta_renda",
        descricao="Produto de investimento (CDB/fundo); hipótese: atrai perfil maduro.",
    ),
    Offer(
        offer_id="renegociacao",
        nome="Plano de Renegociação",
        publico_alvo="negativado",
        descricao="Renegociação de dívida; apenas para indício de inadimplência (default=yes).",
    ),
)

OFFER_IDS: tuple[str, ...] = tuple(o.offer_id for o in OFFER_CATALOG)


def build_offer_catalog() -> pd.DataFrame:
    """Retorna o catálogo de ofertas como DataFrame."""
    return pd.DataFrame([asdict(o) for o in OFFER_CATALOG])


def save_offer_catalog(path: Path | str = OFFER_CATALOG_PATH) -> Path:
    """Salva o catálogo de ofertas em CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    catalog = build_offer_catalog()
    catalog.to_csv(path, index=False)
    logger.info("offer_catalog salvo em %s (%d braços)", path, len(catalog))
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    save_offer_catalog()
