# Relatório técnico - Tech Challenge Fase 2

## 1. Introdução

Este projeto evolui a detecção de risco de sepse da Fase 1 com otimização de hiperparâmetros por Algoritmo Genético, ajuste de threshold e explicações em linguagem natural com LLM ou fallback local.

O sistema é acadêmico e não substitui avaliação médica.

## 2. Problema escolhido

A sepse é uma condição clínica crítica. O principal risco operacional do modelo é o falso negativo: classificar como sem risco um paciente que deveria receber alerta. Por isso, a avaliação prioriza recall, F1-score e redução de falsos negativos.

## 3. Modelo original da Fase 1

O projeto original contém:

- API FastAPI em `__main__.py`;
- dados processados em `data/processed/`;
- artefato em `modelos_salvos/modelo_sepse_sem_tempo_admin.pkl`;
- notebook de análise em `notebook/analise_sepse_2.0.ipynb`.

O artefato original armazena modelo, features, medianas e threshold.

## 4. Estratégia de otimização

O Algoritmo Genético otimiza hiperparâmetros do XGBoost. A fitness do AG prioriza recall e F1-score e penaliza falsos negativos:

```text
fitness = recall * 0.55 + f1_score * 0.35 - fn_penalty * 0.10
```

## 5. Algoritmo Genético

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

O algoritmo implementa população inicial, avaliação, seleção por torneio, crossover uniforme, mutação, elitismo e histórico por geração.

## 6. Experimentos realizados

Foram configurados três experimentos:

| Experimento | População | Gerações | Mutação |
| --- | ---: | ---: | ---: |
| 1 | 10 | 5 | 0.10 |
| 2 | 20 | 8 | 0.20 |
| 3 | 30 | 10 | 0.30 |

Se `reports/ga_experiments_summary.csv` indicar `quick=True`, os resultados presentes são de execução rápida para validação técnica. Para a entrega final, os experimentos completos devem ser executados sem `--quick`.

## 7. Ajuste de threshold

Foi criado `src/tc_fase2/threshold_tuning.py` para escolher o threshold no conjunto de validação, antes da avaliação no teste.

Thresholds avaliados:

```text
0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50
```

Fórmula usada no ajuste:

```text
fitness = recall * 0.50 + f1_score * 0.35 + precision * 0.10 - fn_penalty * 0.05
```

O conjunto de teste não é usado para escolher threshold.

## 8. Resultados

Os resultados finais devem ser gerados pelos comandos sem `--quick`. Quando os arquivos disponíveis forem `quick=True`, eles devem ser tratados apenas como validação técnica do pipeline.

O complemento automático pode ser gerado com:

```bash
python -m src.tc_fase2.update_report_results
```

Arquivo gerado:

- `reports/relatorio_resultados.md`

## 9. Comparação entre baseline e modelo otimizado

`src/tc_fase2/compare_models.py` gera:

- tabela com métricas;
- diferenças absolutas entre baseline e modelo otimizado;
- análise automática do trade-off entre falsos negativos, falsos positivos, precision, recall e F1-score.

## 10. Trade-off do modelo otimizado

Em problemas de sepse, aumentar recall pode elevar falsos positivos. Isso significa mais alertas falsos, mas reduz o risco de deixar de sinalizar pacientes positivos. A interpretação final deve considerar sensibilidade, carga operacional e validação clínica.

## 11. Integração com LLM

`src/tc_fase2/llm_explainer.py` gera explicações em português claro. A LLM:

- usa apenas dados fornecidos;
- não inventa informações ausentes;
- não afirma diagnóstico definitivo;
- reforça que a saída é apoio à decisão clínica;
- recomenda avaliação médica quando houver risco alto;
- informa que risco baixo pelo modelo não elimina avaliação clínica.

## 12. Endpoint `/predict/explain`

A API original foi preservada e recebeu um endpoint adicional:

```text
POST /predict/explain
```

Ele recebe o mesmo payload de `/predict`, calcula probabilidade, classe prevista e retorna uma explicação. Sem chave de API, usa fallback local.

## 13. Exemplo de explicação

Os exemplos positivo e negativo são gerados por:

```bash
python -m src.tc_fase2.llm_explainer --mock
```

Arquivo:

- `reports/llm_explanation_examples.json`

## 14. Escalabilidade, logging e monitoramento

Esta versão acadêmica não implementa escalabilidade automática em nuvem. No entanto, a solução foi estruturada de forma modular, com API FastAPI, Dockerfile, scripts independentes e separação entre dados, modelos, relatórios e testes, permitindo implantação futura em serviços como Render, AWS, Azure ou GCP.

O monitoramento local foi implementado por meio de logs dos experimentos e do treinamento, incluindo início e fim das execuções, configurações utilizadas, métricas, melhor fitness, recall, F1-score, falsos negativos, falsos positivos e tempo de execução.

Assim, a parte de logging e tracking de desempenho foi contemplada no ambiente local, enquanto a escalabilidade automática fica documentada como possibilidade de evolução futura.

Os logs principais são:

- `logs/ga_experiments.log`
- `logs/training.log`

## 15. Testes automatizados

Os testes cobrem AG, métricas, threshold tuning, bloqueio de resultados quick, explicador LLM, script de predição com explicação e endpoints da API.

```bash
pytest
```

## 16. Limitações

- Base desbalanceada.
- Possibilidade de muitos falsos positivos em configurações muito sensíveis.
- Resultados quick não são resultados finais.
- Projeto acadêmico, sem validação clínica real.
- A LLM não faz diagnóstico.
- Uso real exigiria avaliação prospectiva, governança clínica e monitoramento contínuo.

## 17. Conclusão

O projeto agora possui pipeline completo para baseline, AG, ajuste de threshold, treino otimizado, comparação, explicação por LLM/fallback local e relatório de resultados. Os resultados finais devem ser gerados localmente com os comandos sem `--quick`.
