# Detecção de Sepse com Machine Learning - Tech Challenge Fase 2

Projeto acadêmico para apoio à triagem de risco de sepse. A Fase 2 evolui a base da Fase 1 com otimização de hiperparâmetros por Algoritmo Genético, ajuste de threshold em validação e explicações em linguagem natural com LLM ou fallback local.

Este projeto não substitui avaliação médica. A LLM apenas explica a saída do modelo preditivo e não emite diagnóstico definitivo.

## Estrutura

```text
.
|-- __main__.py                         # API FastAPI original + /predict/explain
|-- data/processed/                     # dados processados da Fase 1
|-- modelos_salvos/                     # modelo original
|-- models/                             # modelo otimizado
|-- reports/                            # métricas, gráficos e relatórios
|-- logs/                               # logs
|-- tests/                              # testes automatizados
`-- src/tc_fase2/
    |-- genetic_algorithm.py
    |-- threshold_tuning.py
    |-- train_baseline.py
    |-- run_ga_experiments.py
    |-- train_optimized_model.py
    |-- compare_models.py
    |-- llm_explainer.py
    |-- predict_and_explain.py
    `-- update_report_results.py
```

## Arquitetura da solução

```mermaid
flowchart TD
    A[Dados processados da Fase 1] --> B[Modelo original]
    A --> C[Algoritmo Genético]
    C --> D[Melhores hiperparâmetros]
    D --> E[Ajuste de threshold em validação]
    E --> F[Modelo otimizado]
    F --> G[Comparação de métricas]
    F --> H[Predição com explicação]
    H --> I[LLM ou fallback local]
    G --> J[Relatórios e notebook]
```

## Ambiente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para usar LLM real, configure `OPENAI_API_KEY`. Sem chave, o sistema usa fallback local.

PowerShell:

```powershell
$env:OPENAI_API_KEY="sua_chave"
```

## Execução rápida de teste

Use apenas para validar o fluxo técnico. Resultados `quick=True` não devem ser usados como resultado final da entrega.

```bash
python -m src.tc_fase2.train_baseline --quick
python -m src.tc_fase2.run_ga_experiments --quick
python -m src.tc_fase2.train_optimized_model --quick --allow-quick-results
python -m src.tc_fase2.compare_models
python -m src.tc_fase2.llm_explainer --mock
python -m src.tc_fase2.predict_and_explain
pytest
```

O treino otimizado bloqueia automaticamente hiperparâmetros vindos de `quick=True` quando `--allow-quick-results` não é informado.

## Execução final

Use estes comandos para gerar resultados finais completos:

```bash
python -m src.tc_fase2.train_baseline
python -m src.tc_fase2.run_ga_experiments
python -m src.tc_fase2.train_optimized_model
python -m src.tc_fase2.compare_models
python -m src.tc_fase2.llm_explainer --mock
python -m src.tc_fase2.predict_and_explain
python -m src.tc_fase2.update_report_results
pytest
```

Saídas principais:

- `reports/baseline_metrics.json`
- `reports/ga_experiments_summary.csv`
- `reports/threshold_tuning.csv`
- `reports/best_threshold.json`
- `models/optimized_model.pkl`
- `reports/optimized_metrics.json`
- `reports/model_comparison.md`
- `reports/predict_and_explain_example.json`
- `reports/relatorio_resultados.md`

## Ajuste de threshold

O modelo otimizado não aplica diretamente um threshold fixo no teste. O script `train_optimized_model.py`:

1. treina um modelo com treino;
2. calcula probabilidades na validação;
3. testa thresholds de `0.05` a `0.50`;
4. escolhe o melhor por fitness;
5. treina o modelo final com treino + validação;
6. aplica o threshold escolhido ao teste.

A fórmula documentada é:

```text
fitness = recall * 0.50 + f1_score * 0.35 + precision * 0.10 - fn_penalty * 0.05
```

O teste nunca é usado para escolher threshold.

## Interpretação dos resultados

Em sepse, recall e falsos negativos são mais importantes que accuracy isolada. Um falso negativo pode deixar de sinalizar um paciente em risco.

O trade-off esperado é:

- recall maior tende a reduzir falsos negativos;
- falsos positivos podem subir quando o modelo fica mais sensível;
- precision pode cair se houver muitos alertas falsos;
- F1-score ajuda a observar o equilíbrio entre precision e recall;
- a decisão final sempre depende de avaliação clínica.

## Escalabilidade e uso operacional

A solução foi organizada em módulos para facilitar manutenção e evolução. Em um cenário operacional, a API pode carregar o modelo otimizado salvo em `models/optimized_model.pkl`, aplicar o threshold ajustado em validação e expor a predição por meio dos endpoints `/predict` e `/predict/explain`.

Para uso real, ainda seriam necessários governança clínica, validação externa, monitoramento contínuo de desempenho, controle de drift dos dados e auditoria das explicações geradas pela LLM ou pelo fallback local.

## API

Rodar API:

```bash
python __main__.py
```

Endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /predict/explain`
- `POST /reload`

A API tenta carregar primeiro `models/optimized_model.pkl`. Caso esse arquivo não esteja disponível, utiliza o modelo original em `modelos_salvos/`.

## Endpoint com explicação

`POST /predict/explain` recebe o mesmo payload de `/predict` e retorna:

- `probabilidade_sepse`
- `threshold_utilizado`
- `predicao`
- `classe_predita`
- `explicacao`
- `modo_explicacao`

Sem `OPENAI_API_KEY`, a explicação usa template local seguro.

## Exemplo de payload

```json
{
  "features": {
    "HR": 112,
    "Temp": 38.4,
    "Resp": 28,
    "MAP": 58,
    "Lactate": 3.1,
    "WBC": 16
  },
  "threshold": 0.12
}
```

Features ausentes são preenchidas com medianas do treino quando disponíveis.

## Relatórios

- `reports/relatorio_tecnico.md`: relatório técnico inicial.
- `reports/relatorio_resultados.md`: complemento gerado automaticamente com métricas disponíveis.

Se os CSV/JSON atuais estiverem marcados com `quick=True`, eles representam apenas validação técnica. Rode a execução final para preencher métricas finais reais.

## Notebook da Fase 2

O notebook novo está em `notebook/tech_challenge_fase2_resultados.ipynb`.

Ele apresenta os resultados finais da Fase 2 e não executa treinamento pesado. Também não roda novamente o Algoritmo Genético. O objetivo é carregar os arquivos da pasta `reports/` e apresentar a análise de forma visual, didática e adequada para revisão da banca.

O notebook contém tabelas, explicações em Markdown e gráficos simples em `matplotlib` para facilitar o entendimento de:

- baseline da Fase 1;
- três experimentos do Algoritmo Genético;
- melhor experimento e hiperparâmetros;
- ajuste de threshold em validação;
- comparação entre modelo original e modelo otimizado;
- trade-off entre recall, falsos negativos, falsos positivos, precision e F1-score;
- exemplos de explicação com LLM ou fallback local.
