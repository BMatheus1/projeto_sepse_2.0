# Deteccao de Sepse com Machine Learning - Tech Challenge Fase 2

Projeto academico para apoio a triagem de risco de sepse. A Fase 2 evolui a base da Fase 1 com otimizacao de hiperparametros por Algoritmo Genetico, ajuste de threshold em validacao e explicacoes em linguagem natural com LLM ou fallback local.

Este projeto nao substitui avaliacao medica. A LLM apenas explica a saida do modelo preditivo e nao emite diagnostico definitivo.

## Estrutura

```text
.
|-- __main__.py                         # API FastAPI original + /predict/explain
|-- data/processed/                     # dados processados da Fase 1
|-- modelos_salvos/                     # modelo original
|-- models/                             # modelo otimizado
|-- reports/                            # metricas, graficos e relatorios
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

## Execucao rapida de teste

Use apenas para validar o fluxo tecnico. Resultados `quick=True` nao devem ser usados como resultado final da entrega.

```bash
python -m src.tc_fase2.train_baseline --quick
python -m src.tc_fase2.run_ga_experiments --quick
python -m src.tc_fase2.train_optimized_model --quick --allow-quick-results
python -m src.tc_fase2.compare_models
python -m src.tc_fase2.llm_explainer --mock
python -m src.tc_fase2.predict_and_explain
pytest
```

O treino otimizado bloqueia automaticamente hiperparametros vindos de `quick=True` quando `--allow-quick-results` nao e informado.

## Execucao final

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

Saidas principais:

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

O modelo otimizado nao aplica diretamente um threshold fixo no teste. O script `train_optimized_model.py`:

1. treina um modelo com treino;
2. calcula probabilidades na validacao;
3. testa thresholds de `0.05` a `0.50`;
4. escolhe o melhor por fitness;
5. treina o modelo final com treino + validacao;
6. aplica o threshold escolhido ao teste.

A formula documentada e:

```text
fitness = recall * 0.50 + f1_score * 0.35 + precision * 0.10 - fn_penalty * 0.05
```

O teste nunca e usado para escolher threshold.

## Interpretacao dos resultados

Em sepse, recall e falsos negativos sao mais importantes que accuracy isolada. Um falso negativo pode deixar de sinalizar um paciente em risco.

O trade-off esperado e:

- recall maior tende a reduzir falsos negativos;
- falsos positivos podem subir quando o modelo fica mais sensivel;
- precision pode cair se houver muitos alertas falsos;
- F1-score ajuda a observar equilibrio entre precision e recall;
- a decisao final sempre depende de avaliacao clinica.

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

## Endpoint com explicacao

`POST /predict/explain` recebe o mesmo payload de `/predict` e retorna:

- `probabilidade_sepse`
- `threshold_utilizado`
- `predicao`
- `classe_predita`
- `explicacao`
- `modo_explicacao`

Sem `OPENAI_API_KEY`, a explicacao usa template local seguro.

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

Features ausentes sao preenchidas com medianas do treino quando disponiveis.

## Relatorios

- `reports/relatorio_tecnico.md`: relatorio tecnico inicial.
- `reports/relatorio_resultados.md`: complemento gerado automaticamente com metricas disponiveis.

Se os CSV/JSON atuais estiverem marcados com `quick=True`, eles representam apenas validacao tecnica. Rode a execucao final para preencher metricas finais reais.
