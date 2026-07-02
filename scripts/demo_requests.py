#!/usr/bin/env python
"""Demo script — sobe a API e faz 5 requisições de teste (E5).

Uso: `python scripts/demo_requests.py`
Sobe a API em background, faz requests POST /decide, valida responses, exporta log.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import requests

API_URL = "http://0.0.0.0:8000"
TIMEOUT = 30  # segundos para a API iniciar

# Clientes de teste (diversos segmentos)
TEST_CLIENTS = [
    {
        "client_id": 1,
        "age": 28,
        "job": "student",
        "default": "no",
        "loan": "no",
        "housing": "no",
        "segment": "jovem_baixa_renda",
        "propensity_score": 0.45,
    },
    {
        "client_id": 2,
        "age": 55,
        "job": "management",
        "default": "no",
        "loan": "yes",
        "housing": "yes",
        "segment": "maduro_alta_renda",
        "propensity_score": 0.72,
    },
    {
        "client_id": 3,
        "age": 42,
        "job": "services",
        "default": "yes",
        "loan": "yes",
        "housing": "yes",
        "segment": "negativado",
        "propensity_score": 0.25,
    },
    {
        "client_id": 4,
        "age": 35,
        "job": "admin.",
        "default": "no",
        "loan": "no",
        "housing": "yes",
        "segment": "geral",
        "propensity_score": 0.55,
    },
    {
        "client_id": 5,
        "age": 22,
        "job": "unemployed",
        "default": "no",
        "loan": "no",
        "housing": "no",
        "segment": "jovem_baixa_renda",
        "propensity_score": 0.30,
    },
]


def wait_for_api(url: str, timeout: int = 30) -> bool:
    """Aguarda a API estar pronta (max `timeout` segundos)."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/health", timeout=2)
            if resp.status_code == 200:
                print(f"✓ API pronta em {url}")
                return True
        except requests.exceptions.ConnectionError:
            print("API ainda não disponível; nova tentativa em 1s...")
        time.sleep(1)
    return False


def main():
    """Orquestra a demo."""
    print("=" * 70)
    print("BankMarketing Demo — E5 (Matheus)")
    print("=" * 70)
    print()

    # 1. Limpar log anterior
    print("1. Limpando log de decisões anterior...")
    try:
        requests.post(f"{API_URL}/admin/clear-log", timeout=2)
        print("   ✓ Log limpo")
    except requests.exceptions.ConnectionError:
        print("   (API não está rodando ainda — continuar)")

    # 2. Sober a API em background
    print("\n2. Iniciando API em background...")
    api_process = subprocess.Popen(
        ["python", "-m", "uvicorn", "bankmarketing.app:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("   Processo spawned (PID: {})".format(api_process.pid))

    # 3. Aguardar API
    print("\n3. Aguardando API estar pronta (max {} segundos)...".format(TIMEOUT))
    if not wait_for_api(API_URL, timeout=TIMEOUT):
        print("   ✗ API não iniciou a tempo!")
        api_process.terminate()
        return 1

    # 4. Fazer requests de teste
    print("\n4. Enviando {} requisições POST /decide...".format(len(TEST_CLIENTS)))
    print()

    results = []
    for i, client in enumerate(TEST_CLIENTS, 1):
        try:
            resp = requests.post(
                f"{API_URL}/decide",
                json=client,
                params={"policy": "baseline"},
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()

            results.append(data)

            print(
                f"   [{i}] Client {client['client_id']:2d} (age={client['age']:2d}, seg={client['segment']:20s})"
            )
            print(f"       → Oferta: {data['offer_name']:25s} (conf={data['confidence']:.2f})")
            print(f"       → Decision ID: {data['decision_id']}")

        except requests.exceptions.RequestException as e:
            print(f"   [{i}] ✗ Erro: {e}")
            results.append(None)

    # 5. Recuperar log
    print("\n5. Recuperando log de decisões...")
    try:
        resp = requests.get(f"{API_URL}/log?limit=20", timeout=5)
        resp.raise_for_status()
        log_data = resp.json()
        print(f"   ✓ {log_data['count']} decisões registradas no log")
    except requests.exceptions.RequestException as e:
        print(f"   ✗ Erro ao recuperar log: {e}")

    # 6. Exportar log para Parquet
    print("\n6. Exportando log para Parquet...")
    try:
        from bankmarketing.decision_log import export_log_to_parquet

        path = export_log_to_parquet()
        print(f"   ✓ Log exportado para {path}")
    except Exception as e:
        print(f"   ✗ Erro ao exportar: {e}")

    # 7. Resumo
    print("\n" + "=" * 70)
    print("Demo concluída com sucesso!")
    print("=" * 70)
    print()
    print(f"  Requisições bem-sucedidas: {sum(1 for r in results if r)}/{len(results)}")
    print(f"  Ofertas recomendadas: {len(results)} decisões")
    print()
    print("  Próximos passos:")
    print("    - Adryen: implementar ThompsonSamplingPolicy em E3")
    print("    - Bertelli: avaliar com golden set (E4)")
    print("    - Matheus: integrar LLM para explicações (E5.5)")
    print()

    # 8. Parar a API
    print("Parando API...")
    api_process.terminate()
    api_process.wait(timeout=5)
    print("✓ API parada")

    return 0


if __name__ == "__main__":
    sys.exit(main())
