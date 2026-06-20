# Geração da camada sintética (E2)

> Como a `modeling_table` (E1, base factual de propensão) é enriquecida com a
> camada de experimentação adaptativa que alimenta o bandit (E3).
> Código: [`src/bankmarketing/synthetic.py`](../src/bankmarketing/synthetic.py).
> Testes: [`tests/test_synthetic.py`](../tests/test_synthetic.py).

## Por que uma camada sintética

A base Bank Marketing registra apenas **um** desfecho factual (`y` = assinou
depósito a prazo). Um bandit precisa de **vários braços**, da **resposta a cada
braço** e de **recompensa atrasada** — nada disso existe na base. A E2 fabrica
essa camada de forma controlada, sem que o gerador "entregue a resposta" (ver
[Ataque ao gerador](#ataque-ao-gerador)).

## Artefatos gerados

Todos em `data/synthetic_enrichment/`, regeneráveis por
`python -m bankmarketing.synthetic` (seed padrão = 42).

### 1. `offer_catalog.csv` — catálogo de braços

| coluna | descrição |
|---|---|
| `offer_id` | código do braço (`sem_oferta`, `cartao_credito`, `investimento`, `renegociacao`) |
| `nome` | rótulo legível |
| `publico_alvo` | segmento-alvo da hipótese de negócio |
| `descricao` | descrição da oferta |

`sem_oferta` é o **controle** (não apresentar oferta).

### 2. `offer_events.parquet` — um evento por cliente

| coluna | tipo | descrição |
|---|---|---|
| `event_id` | int | id do evento (0..N-1) |
| `client_idx` | int | índice da linha em `modeling_table` (join com contexto da E1) |
| `segmento` | str | segmento sintético do cliente |
| `propensity_score` | float | score de propensão da E1 (contexto para a E3) |
| `offer_id` | str | braço apresentado (logging policy) |
| `logging_prob` | float | probabilidade de logging do braço (0.25, uniforme) |
| `clicou` | int | resposta imediata (0/1) |

**Granularidade:** uma oferta por cliente (41.188 eventos), fiel à semântica da
base. **Logging policy:** braço sorteado **uniformemente** entre os 4
(`logging_prob = 0.25`) — garante exploração total para o replay offline da E3 e
viabiliza correção por propensão (IPS) na avaliação.

### 3. `delayed_rewards.parquet` — recompensa atrasada

| coluna | tipo | descrição |
|---|---|---|
| `event_id` | int | join com `offer_events` |
| `reward` | int | conversão (0/1) — a recompensa que o bandit otimiza |
| `delay_days` | int | dias até a recompensa materializar (0 se não converteu) |

Funil em dois estágios: **clique imediato** (`offer_events.clicou`) → **conversão
atrasada** (`delayed_rewards.reward`). A recompensa só ocorre se houve clique e
chega após `delay_days` ~ `1 + Poisson(14)` (média ≈ 15 dias). Isso força a E3 a
tratar *delayed feedback* e cold-start.

## Segmentação (proxies de capacidade financeira)

A base **não tem `balance`** (saldo). Usamos proxies, conforme o `PLANO.md`:

| segmento | regra |
|---|---|
| `negativado` | `default == "yes"` (prioridade máxima) |
| `jovem_baixa_renda` | `age ≤ 35` e `job ∈ {student, unemployed, blue-collar, services}` |
| `maduro_alta_renda` | `age ≥ 50` e `job ∈ {management, retired, self-employed, admin.}` e `default == "no"` |
| `geral` | catch-all (demais clientes) |

Distribuição obtida (seed 42): `geral` 31.510 · `jovem_baixa_renda` 6.492 ·
`maduro_alta_renda` 3.183 · `negativado` **3**.

## Processo gerador de recompensa (DGP)

Probabilidade de clique de cada evento:

```
p_click = BASE_CLICK[segmento][braço]
        + BETA_PROPENSITY * (propensity_score - média)   # peso modesto (0.15)
        + ruído ~ Normal(0, 0.03)                          # anti-trivial
p_click = clip(p_click, 0, 0.95);  sem_oferta => 0
clicou  ~ Bernoulli(p_click)
reward  = clicou & Bernoulli(0.5)                          # conversão | clique
```

Matriz-base de clique por (segmento, braço) — codifica as hipóteses de negócio:

| segmento | sem_oferta | cartao_credito | investimento | renegociacao |
|---|---|---|---|---|
| jovem_baixa_renda | 0.00 | **0.16** | 0.05 | 0.04 |
| maduro_alta_renda | 0.00 | 0.06 | **0.18** | 0.03 |
| negativado | 0.00 | 0.03 | 0.03 | **0.15** |
| geral | 0.00 | 0.09 | 0.09 | 0.04 |

**Hipóteses testadas:** jovem/baixa renda → cartão; maduro/alta renda →
investimento; negativado → renegociação (e nunca crédito). Recompensa média
observada (seed 42) confirma os picos esperados por linha: jovem→cartão 0.082,
maduro→investimento 0.098.

### Por que não é trivial

- **Ruído** gaussiano por evento impede que o braço determine a recompensa.
- **Sobreposição**: o segmento `geral` concentra a maioria e tem respostas
  parecidas entre braços.
- **Sinal contínuo**: o `propensity_score` (real, da E1) entra no DGP com peso
  modesto, conectando o sintético ao dado factual sem copiá-lo.

## Ataque ao gerador

DoD da E2: treinar um classificador `reward ~ braço` **sem contexto**. Se a AUC
fosse ≈ 1.0, o gerador teria embutido a resposta. Excluímos o controle
`sem_oferta` (reward 0 estrutural) e atacamos as 3 ofertas reais:

| modelo | AUC |
|---|---|
| só o braço (`offer_id`) | **0.586** |
| braço + segmento + propensão | **0.626** |

Conclusão: o braço sozinho **não** prevê a recompensa (longe de 1.0), e o
contexto agrega poder preditivo — o gerador é não-trivial e dependente de
contexto, como exigido. Verificado em
`tests/test_synthetic.py::test_ataque_ao_gerador_nao_e_trivial`.

## Seeds e reprodutibilidade

- Seed padrão **42** (`DEFAULT_SEED`), via `numpy.random.default_rng`.
- `generate_delayed_rewards` usa `seed + 1` para descorrelacionar do clique.
- Teste `test_geracao_reprodutivel_com_mesma_seed` garante saída idêntica.

## Limitações conhecidas

- **Segmento `negativado` quase vazio (3 clientes).** Na variante
  `bank-additional-full.csv`, `default == "yes"` é degenerado (3 de 41.188; o
  restante é `no`/`unknown`). Logo, a hipótese "negativado → renegociação" **não
  é exercitável a partir da frequência da base**, embora o braço `renegociacao` e
  seu DGP estejam implementados. Mitigações possíveis (decisão de time, exige PR
  no `PLANO.md`): (a) ampliar o proxy de "indício de inadimplência" (ex.:
  `loan == "yes"` ou `default == "unknown"`); (b) manter a definição e cobrir o
  caso via **golden set adversarial** (E4), independente da frequência na base.
- DGP linear nos efeitos de segmento/propensão; sem interações de ordem superior.
- Recompensa binária (conversão). Versão monetária (R$) fica para um possível v1
  caso a FinOps/E8 precise de ROI por oferta.
