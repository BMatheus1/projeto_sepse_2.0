# Deteccao de Sepse com Machine Learning - Tech Challenge Fase 2

Projeto academico de Machine Learning para apoio a triagem de risco de sepse. A Fase 2 evolui o modelo da Fase 1 com otimizacao de hiperparametros por Algoritmo Genetico e explicacoes em linguagem natural com LLM.

Importante: este projeto nao usa dados reais identificaveis de pacientes e nao substitui avaliacao medica. A LLM apenas explica a saida do modelo, sem emitir diagnostico definitivo.

## Problema

A sepse e uma condicao clinica critica. Neste contexto, reduzir falsos negativos e mais importante do que maximizar accuracy isolada, pois um falso negativo pode classificar um paciente com risco como sem risco. Por isso, a otimizacao prioriza:

- recall;
- F1-score;
- penalizacao proporcional a falsos negativos.

## Estrutura

```text
.
|-- __main__.py                         # API FastAPI original
|-- data/processed/                     # dados processados da Fase 1
|-- modelos_salvos/                     # modelo original da Fase 1
|-- models/                             # modelo otimizado da Fase 2
|-- notebook/                           # notebook original
|-- reports/                            # metricas, graficos e relatorios
|-- logs/                               # logs de treino e GA
|-- tests/                              # testes automatizados
|-- src/tc_fase2/
|   |-- config.py
|   |-- project_io.py
|   |-- metrics.py
|   |-- genetic_algorithm.py
|   |-- train_baseline.py
|   |-- run_ga_experiments.py
|   |-- train_optimized_model.py
|   |-- compare_models.py
|   `-- llm_explainer.py
`-- requirements.txt
```

## Ambiente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para usar chamada real a LLM, instale tambem o SDK da OpenAI e configure a chave por variavel de ambiente:

```bash
pip install openai
```

PowerShell:

```powershell
$env:OPENAI_API_KEY="sua_chave"
```

CMD:

```cmd
set OPENAI_API_KEY=sua_chave
```

Sem `OPENAI_API_KEY`, o projeto usa fallback local baseado em template.

## API original

```bash
python __main__.py
```

Endpoints principais:

- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /reload`

## Rodar baseline

Avalia o modelo original salvo em `modelos_salvos/modelo_sepse_sem_tempo_admin.pkl`.

```bash
python -m src.tc_fase2.train_baseline
```

Modo rapido:

```bash
python -m src.tc_fase2.train_baseline --quick
```

Saidas:

- `reports/baseline_metrics.json`
- `reports/baseline_confusion_matrix.png`
- `logs/training.log`

## Rodar experimentos do Algoritmo Genetico

Executa 3 configuracoes obrigatorias de GA:

- populacao 10, geracoes 5, mutacao 0.10;
- populacao 20, geracoes 8, mutacao 0.20;
- populacao 30, geracoes 10, mutacao 0.30.

```bash
python -m src.tc_fase2.run_ga_experiments
```

Modo rapido:

```bash
python -m src.tc_fase2.run_ga_experiments --quick
```

Saidas:

- `reports/ga_experiment_1.json`
- `reports/ga_experiment_2.json`
- `reports/ga_experiment_3.json`
- `reports/ga_experiments_summary.csv`
- `reports/ga_fitness_history.csv`
- `logs/ga_experiments.log`

## Treinar modelo otimizado

Depois dos experimentos, treine o modelo final com os melhores hiperparametros encontrados.

```bash
python -m src.tc_fase2.train_optimized_model
```

Modo rapido:

```bash
python -m src.tc_fase2.train_optimized_model --quick
```

Saidas:

- `models/optimized_model.pkl`
- `reports/optimized_metrics.json`
- `reports/optimized_confusion_matrix.png`
- `logs/training.log`

## Comparar modelos

```bash
python -m src.tc_fase2.compare_models
```

Saidas:

- `reports/model_comparison.csv`
- `reports/model_comparison.md`

## Gerar explicacao com LLM ou fallback local

```bash
python -m src.tc_fase2.llm_explainer --mock
```

Saidas:

- `reports/llm_explanation_examples.json`
- `reports/llm_prompt_used.md`

O prompt instrui a LLM a responder em portugues claro, nao inventar dados, usar apenas as informacoes fornecidas, nao afirmar diagnostico definitivo e reforcar que a saida e apoio a decisao clinica.

## Testes

```bash
pytest
```

Os testes cobrem:

- populacao inicial valida;
- mutacao dentro dos limites;
- crossover valido;
- fitness numerica;
- prompt com dados obrigatorios;
- fallback da LLM sem API key;
- calculo basico de metricas.

## Principais resultados

Os resultados quantitativos devem ser gerados localmente pelos comandos acima. O relatorio inicial esta em `reports/relatorio_tecnico.md` e deixa campos marcados quando dependem da execucao real dos experimentos.

## Grupo

- Matheus Brito da Silva rm373928
- Ricardo Pinto rm374174
- Felipe Monay rm366815
- Ari Monteiro rm371705
- Pedro Artur Araujo Pinto rm373866
