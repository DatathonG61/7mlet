"""Testes para E5 (API, contratos, log de auditoria)."""

import pytest
from bankmarketing.contracts import ClientContext, DecisionResponse, VALID_OFFERS
from bankmarketing.decision_log import (
    clear_log,
    count_log,
    log_decision,
    retrieve_log,
)
from bankmarketing.policies import BaselinePolicy, get_policy
from bankmarketing.app import app

# Fixtures
@pytest.fixture
def client_context_jovem():
    """Cliente do segmento jovem_baixa_renda."""
    return ClientContext(
        client_id=1,
        age=28,
        job="student",
        default="no",
        loan="no",
        housing="no",
        segment="jovem_baixa_renda",
        propensity_score=0.45,
    )


@pytest.fixture
def client_context_maduro():
    """Cliente do segmento maduro_alta_renda."""
    return ClientContext(
        client_id=2,
        age=55,
        job="management",
        default="no",
        loan="yes",
        housing="yes",
        segment="maduro_alta_renda",
        propensity_score=0.72,
    )


@pytest.fixture
def client_context_negativado():
    """Cliente negativado."""
    return ClientContext(
        client_id=3,
        age=42,
        job="services",
        default="yes",
        loan="yes",
        housing="yes",
        segment="negativado",
        propensity_score=0.25,
    )


# Testes de contratos
class TestContracts:
    """Valida modelos Pydantic."""

    def test_client_context_valid(self, client_context_jovem):
        """ClientContext válido."""
        assert client_context_jovem.client_id == 1
        assert client_context_jovem.age == 28
        assert client_context_jovem.segment == "jovem_baixa_renda"

    def test_client_context_invalid_age(self):
        """Idade fora do intervalo deve falhar."""
        with pytest.raises(ValueError):
            ClientContext(
                client_id=1,
                age=200,  # inválido
                job="student",
                default="no",
                loan="no",
                housing="no",
                segment="jovem_baixa_renda",
                propensity_score=0.45,
            )

    def test_decision_response_model(self):
        """DecisionResponse válida."""
        resp = DecisionResponse(
            decision_id="uuid-123",
            timestamp="2026-06-23T10:00:00+00:00",
            offer_id="cartao_credito",
            offer_name="Oferta A — Cartão de Crédito",
            policy="baseline",
            rationale="Segmento jovem de baixa renda.",
            confidence=0.85,
        )
        assert resp.offer_id == "cartao_credito"
        assert resp.policy == "baseline"


# Testes de políticas
class TestPolicies:
    """Valida a BaselinePolicy."""

    def test_baseline_policy_jovem(self, client_context_jovem):
        """Jovem deve receber cartão de crédito."""
        policy = BaselinePolicy()
        decision = policy.decide(
            age=client_context_jovem.age,
            segment=client_context_jovem.segment,
            propensity_score=client_context_jovem.propensity_score,
            default=client_context_jovem.default,
        )
        assert decision.offer_id == "cartao_credito"
        assert decision.confidence >= 0.8

    def test_baseline_policy_maduro(self, client_context_maduro):
        """Maduro deve receber investimento."""
        policy = BaselinePolicy()
        decision = policy.decide(
            age=client_context_maduro.age,
            segment=client_context_maduro.segment,
            propensity_score=client_context_maduro.propensity_score,
            default=client_context_maduro.default,
        )
        assert decision.offer_id == "investimento"
        assert decision.confidence >= 0.8

    def test_baseline_policy_negativado(self, client_context_negativado):
        """Negativado deve receber renegociação."""
        policy = BaselinePolicy()
        decision = policy.decide(
            age=client_context_negativado.age,
            segment=client_context_negativado.segment,
            propensity_score=client_context_negativado.propensity_score,
            default=client_context_negativado.default,
        )
        assert decision.offer_id == "renegociacao"
        assert decision.confidence >= 0.9

    def test_get_policy_baseline(self):
        """Instancia BaselinePolicy via get_policy()."""
        policy = get_policy("baseline")
        assert policy.name == "baseline"

    def test_get_policy_invalid(self):
        """Política inválida deve lançar erro."""
        with pytest.raises(ValueError):
            get_policy("invalid_policy")


# Testes de log de auditoria
class TestDecisionLog:
    """Valida o log de decisões em SQLite."""

    @pytest.fixture(autouse=True)
    def cleanup(self):
        """Limpa o log antes e depois de cada teste."""
        clear_log()
        yield
        clear_log()

    def test_log_decision(self, client_context_jovem):
        """Registra uma decisão no log."""
        entry = log_decision(
            client_id=client_context_jovem.client_id,
            age=client_context_jovem.age,
            job=client_context_jovem.job,
            default=client_context_jovem.default,
            segment=client_context_jovem.segment,
            propensity_score=client_context_jovem.propensity_score,
            policy="baseline",
            offer_id="cartao_credito",
            rationale="Teste",
            confidence=0.85,
        )
        assert entry.decision_id is not None
        assert entry.timestamp is not None
        assert entry.offer_id == "cartao_credito"
        assert entry.policy == "baseline"

    def test_count_log(self, client_context_jovem):
        """Conta registros no log."""
        assert count_log() == 0

        log_decision(
            client_id=client_context_jovem.client_id,
            age=client_context_jovem.age,
            job=client_context_jovem.job,
            default=client_context_jovem.default,
            segment=client_context_jovem.segment,
            propensity_score=client_context_jovem.propensity_score,
            policy="baseline",
            offer_id="cartao_credito",
            rationale="Teste",
            confidence=0.85,
        )

        assert count_log() == 1

    def test_retrieve_log(self, client_context_jovem, client_context_maduro):
        """Recupera e filtra registros do log."""
        # Registra 2 decisões
        log_decision(
            client_id=client_context_jovem.client_id,
            age=client_context_jovem.age,
            job=client_context_jovem.job,
            default=client_context_jovem.default,
            segment=client_context_jovem.segment,
            propensity_score=client_context_jovem.propensity_score,
            policy="baseline",
            offer_id="cartao_credito",
            rationale="Teste 1",
            confidence=0.85,
        )

        log_decision(
            client_id=client_context_maduro.client_id,
            age=client_context_maduro.age,
            job=client_context_maduro.job,
            default=client_context_maduro.default,
            segment=client_context_maduro.segment,
            propensity_score=client_context_maduro.propensity_score,
            policy="baseline",
            offer_id="investimento",
            rationale="Teste 2",
            confidence=0.90,
        )

        # Recupera todas
        entries = retrieve_log(limit=10)
        assert len(entries) == 2

        # Filtra por oferta
        entries = retrieve_log(limit=10, offer_id="cartao_credito")
        assert len(entries) == 1
        assert entries[0].offer_id == "cartao_credito"


# Testes da API FastAPI
class TestAPI:
    """Valida endpoints FastAPI."""

    @pytest.fixture
    def api_client(self):
        """Cliente de teste FastAPI."""
        from fastapi.testclient import TestClient

        return TestClient(app)

    def test_health(self, api_client):
        """GET /health retorna ok."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_root(self, api_client):
        """GET / retorna informações."""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data

    def test_decide_baseline(self, api_client, client_context_jovem):
        """POST /decide retorna DecisionResponse válida."""
        response = api_client.post(
            "/decide",
            json=client_context_jovem.model_dump(),
            params={"policy": "baseline"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "decision_id" in data
        assert "offer_id" in data
        assert data["offer_id"] == "cartao_credito"
        assert data["policy"] == "baseline"

    def test_decide_invalid_policy(self, api_client, client_context_jovem):
        """POST /decide com política inválida retorna 422 (Unprocessable Entity)."""
        response = api_client.post(
            "/decide",
            json=client_context_jovem.model_dump(),
            params={"policy": "invalid"},
        )
        assert response.status_code == 422

    def test_log_endpoint(self, api_client, client_context_jovem):
        """GET /log retorna histórico."""
        # Faz uma decisão
        api_client.post(
            "/decide",
            json=client_context_jovem.model_dump(),
            params={"policy": "baseline"},
        )

        # Recupera log
        response = api_client.get("/log?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1
        assert len(data["entries"]) >= 1
