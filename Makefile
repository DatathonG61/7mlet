# Makefile — comandos reproduzíveis do projeto BankMarketing
# Uso: `make <target>`. Requer uv (https://docs.astral.sh/uv/).

.PHONY: help install test lint demo

help:  ## Lista os targets disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Sincroniza o ambiente (cria .venv e instala deps do uv.lock)
	uv sync

test:  ## Roda a suíte de testes
	uv run pytest

lint:  ## Roda o linter (ruff)
	uvx ruff check .

demo:  ## Pipeline ponta a ponta — sobe API + 5 requests de teste (E5)
	@echo ">> [E5] Demo ponta a ponta: validação + API + requests"
	@echo ""
	@echo "1. Validando imports..."
	python -c "import bankmarketing; print('✓ bankmarketing importa')"
	@echo ""
	@echo "2. Rodando testes..."
	python -m pytest -q
	@echo ""
	@echo "3. Iniciando API em http://0.0.0.0:8000..."
	@echo "   (rodando em background por 30 segundos)"
	@echo ""
	python scripts/demo_requests.py
	@echo ""
	@echo "✓ Demo concluída!"
