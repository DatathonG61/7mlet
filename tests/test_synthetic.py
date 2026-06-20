import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from bankmarketing.data import MODELING_TABLE_PATH, load_modeling_table
from bankmarketing.synthetic import (
    OFFER_IDS,
    PROPENSITY_PATH,
    assign_segments,
    build_offer_catalog,
    generate_delayed_rewards,
    generate_offer_events,
    save_offer_catalog,
)

EXPECTED_OFFER_IDS = {"sem_oferta", "cartao_credito", "investimento", "renegociacao"}


def _fake_modeling_table(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Tabela mínima com as colunas que o gerador consome."""
    rng = np.random.default_rng(seed)
    jobs = ["student", "blue-collar", "management", "retired", "admin.", "technician"]
    return pd.DataFrame(
        {
            "age": rng.integers(18, 90, size=n),
            "job": rng.choice(jobs, size=n),
            "default": rng.choice(["no", "unknown", "yes"], size=n, p=[0.8, 0.19, 0.01]),
        }
    )


def test_offer_catalog_has_four_arms():
    catalog = build_offer_catalog()
    assert len(catalog) == 4
    assert set(catalog["offer_id"]) == EXPECTED_OFFER_IDS


def test_offer_ids_match_catalog():
    assert set(OFFER_IDS) == EXPECTED_OFFER_IDS


def test_offer_catalog_columns():
    catalog = build_offer_catalog()
    assert list(catalog.columns) == ["offer_id", "nome", "publico_alvo", "descricao"]


def test_save_offer_catalog(tmp_path):
    path = tmp_path / "offer_catalog.csv"
    save_offer_catalog(path=path)
    assert path.exists()
    reloaded = pd.read_csv(path)
    assert set(reloaded["offer_id"]) == EXPECTED_OFFER_IDS


# --- segmentação ---------------------------------------------------------


def test_assign_segments_negativado_tem_prioridade():
    df = pd.DataFrame(
        {
            "age": [25, 60, 30],
            "job": ["student", "management", "blue-collar"],
            "default": ["yes", "yes", "no"],  # 1 e 2 negativados apesar de idade/job
        }
    )
    seg = assign_segments(df)
    assert seg.tolist() == ["negativado", "negativado", "jovem_baixa_renda"]


def test_assign_segments_valores_validos():
    seg = assign_segments(_fake_modeling_table())
    assert set(seg.unique()).issubset(
        {"jovem_baixa_renda", "maduro_alta_renda", "negativado", "geral"}
    )


# --- eventos e recompensas ----------------------------------------------


def _events_rewards(seed: int = 42):
    df = _fake_modeling_table()
    prop = pd.Series(np.random.default_rng(1).uniform(0, 0.9, size=len(df)), index=df.index)
    events = generate_offer_events(df, prop, seed=seed)
    rewards = generate_delayed_rewards(events, seed=seed)
    return df, events, rewards


def test_offer_events_um_por_cliente_e_schema():
    df, events, _ = _events_rewards()
    assert len(events) == len(df)
    assert list(events.columns) == [
        "event_id",
        "client_idx",
        "segmento",
        "propensity_score",
        "offer_id",
        "logging_prob",
        "clicou",
    ]
    assert set(events["offer_id"]).issubset(set(OFFER_IDS))
    assert events["clicou"].isin([0, 1]).all()


def test_delayed_rewards_schema_e_dependencia_do_clique():
    _, events, rewards = _events_rewards()
    assert list(rewards.columns) == ["event_id", "reward", "delay_days"]
    m = events.merge(rewards, on="event_id")
    # recompensa só existe quando houve clique
    assert (m.loc[m["reward"] == 1, "clicou"] == 1).all()
    # convertidos têm delay > 0; não-convertidos têm delay 0
    assert (m.loc[m["reward"] == 1, "delay_days"] > 0).all()
    assert (m.loc[m["reward"] == 0, "delay_days"] == 0).all()


def test_sem_oferta_nunca_converte():
    _, events, rewards = _events_rewards()
    m = events.merge(rewards, on="event_id")
    assert m.loc[m["offer_id"] == "sem_oferta", "reward"].sum() == 0


def test_geracao_reprodutivel_com_mesma_seed():
    _, ev1, rw1 = _events_rewards(seed=7)
    _, ev2, rw2 = _events_rewards(seed=7)
    pd.testing.assert_frame_equal(ev1, ev2)
    pd.testing.assert_frame_equal(rw1, rw2)


# --- DoD: "ataque ao gerador" -------------------------------------------


def test_ataque_ao_gerador_nao_e_trivial():
    """O braço sozinho não pode prever a recompensa (gerador não-trivial).

    Treina-se um classificador reward~braço SEM contexto: se a AUC fosse ~1.0, o
    gerador teria embutido a resposta. Verifica-se também que adicionar contexto
    (segmento + propensão) melhora a AUC — ou seja, a recompensa depende do contexto.
    """
    try:
        modeling_table = load_modeling_table(MODELING_TABLE_PATH)
        prop = pd.read_parquet(PROPENSITY_PATH)["propensity_score"].reindex(
            modeling_table.index
        )
    except FileNotFoundError:
        pytest.skip("artefatos de E1 ausentes (modeling_table/propensity)")

    events = generate_offer_events(modeling_table, prop, seed=42)
    rewards = generate_delayed_rewards(events, seed=42)
    data = events.merge(rewards, on="event_id")
    # Exclui o controle: sem_oferta tem reward 0 por definição (estrutural, não
    # "resposta embutida"). O ataque justo é entre as ofertas reais.
    data = data[data["offer_id"] != "sem_oferta"].reset_index(drop=True)
    y = data["reward"]

    arm_dummies = pd.get_dummies(data["offer_id"])
    seg_dummies = pd.get_dummies(data["segmento"])
    x_arm = arm_dummies
    x_full = pd.concat([arm_dummies, seg_dummies, data[["propensity_score"]]], axis=1)

    def auc(x):
        x_tr, x_te, y_tr, y_te = train_test_split(
            x, y, test_size=0.3, stratify=y, random_state=0
        )
        clf = LogisticRegression(max_iter=1000).fit(x_tr, y_tr)
        return roc_auc_score(y_te, clf.predict_proba(x_te)[:, 1])

    auc_arm = auc(x_arm)
    auc_full = auc(x_full)

    # Não-trivial: só o braço não explica a recompensa.
    assert auc_arm < 0.70, f"gerador trivial demais: AUC só com braço = {auc_arm:.3f}"
    # O contexto importa: agrega poder preditivo sobre o braço isolado.
    assert auc_full > auc_arm, f"contexto não ajudou: arm={auc_arm:.3f} full={auc_full:.3f}"
