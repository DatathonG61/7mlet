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

import numpy as np
import pandas as pd

from bankmarketing.data import MODELING_TABLE_PATH, PROCESSED_DIR, load_modeling_table

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SYNTH_DIR = ROOT / "data" / "synthetic_enrichment"
OFFER_CATALOG_PATH = SYNTH_DIR / "offer_catalog.csv"
OFFER_EVENTS_PATH = SYNTH_DIR / "offer_events.parquet"
DELAYED_REWARDS_PATH = SYNTH_DIR / "delayed_rewards.parquet"
PROPENSITY_PATH = PROCESSED_DIR / "propensity_scores.parquet"

DEFAULT_SEED = 42


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


# ---------------------------------------------------------------------------
# Segmentação por proxies de capacidade financeira (não há `balance` na base).
# Definições alinhadas ao PLANO.md.
# ---------------------------------------------------------------------------

JOBS_BAIXA_RENDA = frozenset({"student", "unemployed", "blue-collar", "services"})
JOBS_ALTA_RENDA = frozenset({"management", "retired", "self-employed", "admin."})

SEGMENTS = ("jovem_baixa_renda", "maduro_alta_renda", "negativado", "geral")


def assign_segments(df: pd.DataFrame) -> pd.Series:
    """Classifica cada cliente em um segmento sintético (vetorizado).

    Ordem de prioridade: ``negativado`` (default=yes) → ``jovem_baixa_renda``
    → ``maduro_alta_renda`` → ``geral`` (catch-all).
    """
    seg = pd.Series("geral", index=df.index, dtype="object")

    is_jovem = (df["age"] <= 35) & df["job"].isin(JOBS_BAIXA_RENDA)
    is_maduro = (df["age"] >= 50) & df["job"].isin(JOBS_ALTA_RENDA) & (df["default"] == "no")
    is_negativado = df["default"] == "yes"

    seg[is_jovem] = "jovem_baixa_renda"
    seg[is_maduro] = "maduro_alta_renda"
    seg[is_negativado] = "negativado"  # prioridade máxima: sobrescreve os anteriores
    return seg


# ---------------------------------------------------------------------------
# Processo gerador de recompensa (DGP)
# ---------------------------------------------------------------------------
# Probabilidade-base de clique por (segmento, braço). Codifica as hipóteses de
# negócio: jovem→cartão, maduro→investimento, negativado→renegociação.
# `sem_oferta` é controle: clique sempre 0 (nada é apresentado).
BASE_CLICK_PROB: dict[str, dict[str, float]] = {
    "jovem_baixa_renda": {"sem_oferta": 0.0, "cartao_credito": 0.16, "investimento": 0.05, "renegociacao": 0.04},  # noqa: E501
    "maduro_alta_renda": {"sem_oferta": 0.0, "cartao_credito": 0.06, "investimento": 0.18, "renegociacao": 0.03},  # noqa: E501
    "negativado": {"sem_oferta": 0.0, "cartao_credito": 0.03, "investimento": 0.03, "renegociacao": 0.15},  # noqa: E501
    "geral": {"sem_oferta": 0.0, "cartao_credito": 0.09, "investimento": 0.09, "renegociacao": 0.04},  # noqa: E501
}

# Peso (modesto) do score de propensão da E1 no DGP: cliente intrinsecamente mais
# "engajável" responde um pouco melhor a qualquer oferta. Modesto de propósito para
# não dominar o efeito do segmento (que carrega as hipóteses de negócio).
BETA_PROPENSITY = 0.15
# Ruído gaussiano por evento — impede gerador trivial (resposta determinística por braço).
CLICK_NOISE_STD = 0.03
# Conversão (recompensa) condicional ao clique.
P_CONVERT_GIVEN_CLICK = 0.5
# Atraso médio (dias) até a recompensa materializar — delayed rewards.
REWARD_DELAY_MEAN_DAYS = 14


def generate_offer_events(
    modeling_table: pd.DataFrame,
    propensity: pd.Series,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Gera um evento de oferta por cliente (logging policy aleatória uniforme).

    Cada cliente recebe um braço sorteado uniformemente entre os 4 (propensão de
    logging = 0.25 por braço — exploração total para replay offline). O clique é
    amostrado de ``Bernoulli(p)`` com ``p`` função do (segmento, braço), do score de
    propensão e de ruído.

    Args:
        modeling_table: tabela de modelagem (E1).
        propensity: score de propensão por cliente, alinhado por índice (E1).
        seed: semente para reprodutibilidade.

    Returns:
        DataFrame com colunas: ``event_id, client_idx, segmento, propensity_score,
        offer_id, logging_prob, clicou``.
    """
    rng = np.random.default_rng(seed)
    n = len(modeling_table)

    segmento = assign_segments(modeling_table).to_numpy()
    prop = propensity.to_numpy()
    prop_mean = float(prop.mean())

    # Logging policy: braço uniforme entre os 4.
    arm_idx = rng.integers(0, len(OFFER_IDS), size=n)
    offer_id = np.array(OFFER_IDS)[arm_idx]

    base_p = np.array([BASE_CLICK_PROB[s][o] for s, o in zip(segmento, offer_id)])
    noise = rng.normal(0.0, CLICK_NOISE_STD, size=n)
    p_click = base_p + BETA_PROPENSITY * (prop - prop_mean) + noise
    p_click = np.clip(p_click, 0.0, 0.95)
    p_click[offer_id == "sem_oferta"] = 0.0  # controle nunca gera clique

    clicou = (rng.random(n) < p_click).astype(int)

    return pd.DataFrame(
        {
            "event_id": np.arange(n),
            "client_idx": modeling_table.index.to_numpy(),
            "segmento": segmento,
            "propensity_score": prop,
            "offer_id": offer_id,
            "logging_prob": 1.0 / len(OFFER_IDS),
            "clicou": clicou,
        }
    )


def generate_delayed_rewards(
    events: pd.DataFrame,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Gera a recompensa (conversão) atrasada para cada evento.

    A recompensa só pode ocorrer se houve clique; quando ocorre, materializa após
    ``delay_days`` (Poisson). Eventos sem conversão têm ``reward=0`` e ``delay_days=0``.

    Args:
        events: saída de :func:`generate_offer_events`.
        seed: semente (deslocada internamente para não correlacionar com o clique).

    Returns:
        DataFrame com colunas: ``event_id, reward, delay_days``.
    """
    rng = np.random.default_rng(seed + 1)
    clicou = events["clicou"].to_numpy()

    convertido = (clicou == 1) & (rng.random(len(events)) < P_CONVERT_GIVEN_CLICK)
    reward = convertido.astype(int)

    delay_days = np.zeros(len(events), dtype=int)
    n_conv = int(convertido.sum())
    delay_days[convertido] = 1 + rng.poisson(REWARD_DELAY_MEAN_DAYS, size=n_conv)

    return pd.DataFrame(
        {
            "event_id": events["event_id"].to_numpy(),
            "reward": reward,
            "delay_days": delay_days,
        }
    )


def generate_all(seed: int = DEFAULT_SEED) -> dict[str, Path]:
    """Gera e salva os três artefatos sintéticos a partir dos dados de E1.

    Returns:
        Mapa ``{nome: caminho}`` dos arquivos escritos.
    """
    modeling_table = load_modeling_table(MODELING_TABLE_PATH)
    propensity = pd.read_parquet(PROPENSITY_PATH)["propensity_score"]
    propensity = propensity.reindex(modeling_table.index)

    events = generate_offer_events(modeling_table, propensity, seed=seed)
    rewards = generate_delayed_rewards(events, seed=seed)

    catalog_path = save_offer_catalog()
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    events.to_parquet(OFFER_EVENTS_PATH, index=False)
    rewards.to_parquet(DELAYED_REWARDS_PATH, index=False)

    logger.info(
        "Sintéticos gerados (seed=%d): %d eventos, taxa de clique=%.3f, "
        "taxa de conversão=%.3f",
        seed,
        len(events),
        float(events["clicou"].mean()),
        float(rewards["reward"].mean()),
    )
    return {
        "offer_catalog": catalog_path,
        "offer_events": OFFER_EVENTS_PATH,
        "delayed_rewards": DELAYED_REWARDS_PATH,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    generate_all()
