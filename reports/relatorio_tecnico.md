# Relatorio tecnico - Tech Challenge Fase 2

## 1. Introducao

Este projeto evolui a deteccao de risco de sepse da Fase 1 com otimizacao de hiperparametros por Algoritmo Genetico, ajuste de threshold e explicacoes em linguagem natural com LLM ou fallback local.

O sistema e academico e nao substitui avaliacao medica.

## 2. Problema escolhido

A sepse e uma condicao clinica critica. O principal risco operacional do modelo e o falso negativo: classificar como sem risco um paciente que deveria receber alerta. Por isso, a avaliacao prioriza recall, F1-score e reducao de falsos negativos.

## 3. Modelo original da Fase 1

O projeto original contem:

- API FastAPI em `__main__.py`;
- dados processados em `data/processed/`;
- artefato em `modelos_salvos/modelo_sepse_sem_tempo_admin.pkl`;
- notebook de analise em `notebook/analise_sepse_2.0.ipynb`.

O artefato original armazena modelo, features, medianas e threshold.

## 4. Estrategia de otimizacao

O Algoritmo Genetico otimiza hiperparametros do XGBoost. A fitness do GA prioriza recall e F1-score e penaliza falsos negativos:

```text
fitness = recall * 0.55 + f1_score * 0.35 - fn_penalty * 0.10
```

## 5. Algoritmo genetico

Genes avaliados:

- `max_depth`
- `learning_rate`
- `n_estimators`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`
- `reg_alpha`
- `reg_lambda`

O algoritmo implementa populacao inicial, avaliacao, selecao por torneio, crossover uniforme, mutacao, elitismo e historico por geracao.

## 6. Experimentos realizados

Foram configurados tres experimentos:

| Experimento | Populacao | Geracoes | Mutacao |
| --- | ---: | ---: | ---: |
| 1 | 10 | 5 | 0.10 |
| 2 | 20 | 8 | 0.20 |
| 3 | 30 | 10 | 0.30 |

Se `reports/ga_experiments_summary.csv` indicar `quick=True`, os resultados presentes sao de execucao rapida para validacao tecnica. Para a entrega final, executar os experimentos completos sem `--quick`.

## 7. Ajuste de threshold

Foi criado `src/tc_fase2/threshold_tuning.py` para escolher threshold no conjunto de validacao, antes da avaliacao no teste.

Thresholds avaliados:

```text
0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
```

Formula usada no ajuste:

```text
fitness = recall * 0.50 + f1_score * 0.35 + precision * 0.10 - fn_penalty * 0.05
```

O conjunto de teste nao e usado para escolher threshold.

## 8. Resultados

Os resultados finais devem ser gerados pelos comandos sem `--quick`. Quando os arquivos disponiveis forem `quick=True`, eles devem ser tratados apenas como validacao tecnica do pipeline.

O complemento automatico pode ser gerado com:

```bash
python -m src.tc_fase2.update_report_results
```

Arquivo gerado:

- `reports/relatorio_resultados.md`

## 9. Comparacao entre baseline e otimizado

`src/tc_fase2/compare_models.py` gera:

- tabela com metricas;
- diferencas absolutas entre baseline e otimizado;
- analise automatica do trade-off entre falsos negativos, falsos positivos, precision, recall e F1-score.

## 10. Trade-off do modelo otimizado

Em problemas de sepse, aumentar recall pode elevar falsos positivos. Isso significa mais alertas falsos, mas reduz o risco de deixar de sinalizar pacientes positivos. A interpretacao final deve considerar sensibilidade, carga operacional e validacao clinica.

## 11. Integracao com LLM

`src/tc_fase2/llm_explainer.py` gera explicacoes em portugues claro. A LLM:

- usa apenas dados fornecidos;
- nao inventa informacoes ausentes;
- nao afirma diagnostico definitivo;
- reforca que a saida e apoio a decisao clinica;
- recomenda avaliacao medica quando houver risco alto;
- informa que risco baixo pelo modelo nao elimina avaliacao clinica.

## 12. Endpoint `/predict/explain`

A API original foi preservada e recebeu um endpoint adicional:

```text
POST /predict/explain
```

Ele recebe o mesmo payload de `/predict`, calcula probabilidade, classe prevista e retorna uma explicacao. Sem chave de API, usa fallback local.

## 13. Exemplo de explicacao

Os exemplos positivo e negativo sao gerados por:

```bash
python -m src.tc_fase2.llm_explainer --mock
```

Arquivo:

- `reports/llm_explanation_examples.json`

## 14. Logging e monitoramento

Os logs principais sao:

- `logs/ga_experiments.log`
- `logs/training.log`

Eles registram inicio/fim, configuracao, geracao atual, melhor fitness, metricas e tempo de execucao.

## 15. Testes automatizados

Os testes cobrem GA, metricas, threshold tuning, bloqueio de resultados quick, explicador LLM, script de predicao com explicacao e endpoints da API.

```bash
pytest
```

## 16. Limitacoes

- Base desbalanceada.
- Possibilidade de muitos falsos positivos em configuracoes muito sensiveis.
- Resultados quick nao sao resultados finais.
- Projeto academico, sem validacao clinica real.
- A LLM nao faz diagnostico.
- Uso real exigiria avaliacao prospectiva, governanca clinica e monitoramento continuo.

## 17. Conclusao

O projeto agora possui pipeline completo para baseline, GA, ajuste de threshold, treino otimizado, comparacao, explicacao por LLM/fallback e relatorio de resultados. Os resultados finais devem ser gerados localmente com os comandos sem `--quick`.
