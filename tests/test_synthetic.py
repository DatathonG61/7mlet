import pandas as pd

from bankmarketing.synthetic import (
    OFFER_IDS,
    build_offer_catalog,
    save_offer_catalog,
)

EXPECTED_OFFER_IDS = {"sem_oferta", "cartao_credito", "investimento", "renegociacao"}


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
