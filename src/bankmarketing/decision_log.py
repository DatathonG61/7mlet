"""Log de auditoria de decisões (E5).

Registra cada decisão tomada pelo sistema adaptativo, com rastreabilidade completa:
- Cliente (contexto: age, job, default, segmento)
- Oferta apresentada
- Política que gerou a decisão (baseline, thompson, etc.)
- Clique do cliente (resposta imediata)
- Conversão + delay (recompensa atrasada, sincronizada offline)
- Motivo da recomendação (rationale)

O log é armazenado em SQLite local (será Cosmos DB em produção no Azure).
Exportável para Parquet para replay offline (E4, E7).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bankmarketing.contracts import DecisionLogEntry
from bankmarketing.synthetic import OFFER_CATALOG, OFFER_IDS

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "decision_log.db"
LOG_EXPORT_PATH = ROOT / "data" / "decision_log.parquet"


def _ensure_db():
    """Cria o schema SQLite se não existir."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_log (
            decision_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            client_id INTEGER NOT NULL,
            age INTEGER NOT NULL,
            job TEXT NOT NULL,
            "default" TEXT NOT NULL,
            segment TEXT NOT NULL,
            propensity_score REAL NOT NULL,
            policy TEXT NOT NULL,
            offer_id TEXT NOT NULL,
            offer_name TEXT NOT NULL,
            rationale TEXT NOT NULL,
            confidence REAL NOT NULL,
            clicked INTEGER NOT NULL DEFAULT 0,
            reward INTEGER NOT NULL DEFAULT 0,
            delay_days INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )

    # Índices para buscas rápidas
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON decision_log(timestamp DESC)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_policy ON decision_log(policy)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_offer ON decision_log(offer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_segment ON decision_log(segment)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_client ON decision_log(client_id)")

    conn.commit()
    conn.close()

    logger.info("Decision log DB inicializado em %s", DB_PATH)


def log_decision(
    client_id: int,
    age: int,
    job: str,
    default: str,
    segment: str,
    propensity_score: float,
    policy: str,
    offer_id: str,
    rationale: str,
    confidence: float = 0.5,
    clicked: int = 0,
) -> DecisionLogEntry:
    """Registra uma decisão no log de auditoria.

    Args:
        client_id: índice/ID do cliente (chave na modeling_table).
        age: idade do cliente.
        job: profissão do cliente.
        default: status de inadimplência ("yes" ou "no").
        segment: segmento sintético (jovem_baixa_renda, maduro_alta_renda, negativado, geral).
        propensity_score: score de propensão da E1 (0–1).
        policy: política que decidiu (ex.: "baseline", "thompson_sampling").
        offer_id: ID da oferta (deve estar em OFFER_IDS).
        rationale: explicação em texto livre da recomendação.
        confidence: confiança da recomendação (0–1).
        clicked: 1 se cliente clicou, 0 caso contrário. Sincronizado com delayed_rewards offline.

    Returns:
        DecisionLogEntry registrada (contém decision_id e timestamp).

    Raises:
        ValueError: se offer_id não está no catálogo.
    """
    if offer_id not in OFFER_IDS:
        raise ValueError(
            f"offer_id '{offer_id}' não está no catálogo. Válidos: {OFFER_IDS}"
        )

    offer_name = next(o.nome for o in OFFER_CATALOG if o.offer_id == offer_id)

    decision_id = str(uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO decision_log (
            decision_id, timestamp, client_id, age, job, "default", segment,
            propensity_score, policy, offer_id, offer_name, rationale, confidence,
            clicked, reward, delay_days, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            decision_id,
            timestamp,
            client_id,
            age,
            job,
            default,
            segment,
            propensity_score,
            policy,
            offer_id,
            offer_name,
            rationale,
            confidence,
            clicked,
            0,  # reward
            0,  # delay_days
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    conn.commit()
    conn.close()

    logger.debug(
        "Decision registrada: %s (client_id=%d, offer=%s, policy=%s)",
        decision_id,
        client_id,
        offer_id,
        policy,
    )

    return DecisionLogEntry(
        decision_id=decision_id,
        timestamp=timestamp,
        client_id=client_id,
        age=age,
        job=job,
        default=default,
        segment=segment,
        propensity_score=propensity_score,
        policy=policy,
        offer_id=offer_id,
        offer_name=offer_name,
        rationale=rationale,
        confidence=confidence,
        clicked=clicked,
        reward=0,
        delay_days=0,
    )


def retrieve_log(
    limit: int = 100,
    policy: str | None = None,
    offer_id: str | None = None,
    segment: str | None = None,
    client_id: int | None = None,
) -> list[DecisionLogEntry]:
    """Recupera decisões do log com filtros opcionais.

    Args:
        limit: número máximo de registros a retornar (ordenado por timestamp DESC).
        policy: filtrar por política (ex.: "thompson_sampling").
        offer_id: filtrar por oferta (ex.: "cartao_credito").
        segment: filtrar por segmento (ex.: "jovem_baixa_renda").
        client_id: filtrar por cliente específico.

    Returns:
        Lista de DecisionLogEntry, mais recentes primeiro.
    """
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM decision_log WHERE 1=1"
    params = []

    if policy is not None:
        query += " AND policy = ?"
        params.append(policy)
    if offer_id is not None:
        query += " AND offer_id = ?"
        params.append(offer_id)
    if segment is not None:
        query += " AND segment = ?"
        params.append(segment)
    if client_id is not None:
        query += " AND client_id = ?"
        params.append(client_id)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    entries = [
        DecisionLogEntry(
            decision_id=row["decision_id"],
            timestamp=row["timestamp"],
            client_id=row["client_id"],
            age=row["age"],
            job=row["job"],
            default=row["default"],
            segment=row["segment"],
            propensity_score=row["propensity_score"],
            policy=row["policy"],
            offer_id=row["offer_id"],
            offer_name=row["offer_name"],
            rationale=row["rationale"],
            confidence=row["confidence"],
            clicked=row["clicked"],
            reward=row["reward"],
            delay_days=row["delay_days"],
        )
        for row in rows
    ]

    return entries


def count_log() -> int:
    """Conta o total de decisões no log."""
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM decision_log")
    count = cursor.fetchone()[0]
    conn.close()

    return count


def sync_rewards(event_id_to_reward: dict[str, tuple[int, int]]) -> int:
    """Sincroniza recompensas atrasadas (delayed rewards) no log.

    Chamado offline (E4, E7) após o período de atraso: mapeia
    event_id → (reward, delay_days) e atualiza as entradas correspondentes.

    Args:
        event_id_to_reward: dicionário {decision_id: (reward, delay_days)}.

    Returns:
        Número de linhas atualizadas.
    """
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    n_updated = 0
    for decision_id, (reward, delay_days) in event_id_to_reward.items():
        cursor.execute(
            """
            UPDATE decision_log
            SET reward = ?, delay_days = ?
            WHERE decision_id = ?
            """,
            (reward, delay_days, decision_id),
        )
        n_updated += cursor.rowcount

    conn.commit()
    conn.close()

    logger.info("Rewards sincronizados: %d linhas atualizadas", n_updated)
    return n_updated


def export_log_to_parquet(path: Path | str = LOG_EXPORT_PATH) -> Path:
    """Exporta o log de decisões inteiro para Parquet (para replay offline).

    Args:
        path: caminho de destino.

    Returns:
        Caminho do arquivo exportado.
    """
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM decision_log ORDER BY timestamp DESC", conn
    )
    conn.close()

    if len(df) == 0:
        logger.warning("Log está vazio, não exportando")
        return Path(path)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Garante tipos corretos para Parquet
    schema = pa.schema(
        [
            ("decision_id", pa.string()),
            ("timestamp", pa.string()),
            ("client_id", pa.int64()),
            ("age", pa.int64()),
            ("job", pa.string()),
            ("default", pa.string()),
            ("segment", pa.string()),
            ("propensity_score", pa.float64()),
            ("policy", pa.string()),
            ("offer_id", pa.string()),
            ("offer_name", pa.string()),
            ("rationale", pa.string()),
            ("confidence", pa.float64()),
            ("clicked", pa.int64()),
            ("reward", pa.int64()),
            ("delay_days", pa.int64()),
        ]
    )

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, path)

    logger.info("Decision log exportado para %s (%d decisões)", path, len(df))
    return path


def clear_log() -> None:
    """Limpa o log de decisões (uso: reset para testes)."""
    _ensure_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM decision_log")
    conn.commit()
    conn.close()

    logger.warning("Decision log foi limpado")
