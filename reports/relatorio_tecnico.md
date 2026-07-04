# Relatorio tecnico - Tech Challenge Fase 2

## 1. Introducao

Este relatorio descreve a evolucao do projeto de deteccao de sepse desenvolvido na Fase 1. A Fase 2 adiciona otimizacao de hiperparametros com Algoritmo Genetico e um modulo de explicacao em linguagem natural com LLM ou fallback local.

## 2. Problema escolhido

O problema escolhido e a triagem de risco de sepse com Machine Learning. Em contexto medico, a prioridade e reduzir falsos negativos, pois deixar de sinalizar um paciente em risco pode ter impacto clinico relevante.

## 3. Modelo original da Fase 1

O projeto original possui uma API FastAPI em `__main__.py`, dados processados em `data/processed/`, notebook em `notebook/analise_sepse_2.0.ipynb` e artefato salvo em `modelos_salvos/modelo_sepse_sem_tempo_admin.pkl`.

O artefato original contem um `XGBClassifier`, lista de features, medianas de treino e threshold de validacao.

Resultados do baseline: preencher apos executar `python -m src.tc_fase2.train_baseline`.

## 4. Estrategia de otimizacao

A otimizacao nao usa accuracy como objetivo principal. A funcao fitness combina recall, F1-score e penalizacao proporcional aos falsos negativos.

Formula usada:

```text
fitness = recall * 0.55 + f1_score * 0.35 - penalizacao_fn * 0.10
```

A penalizacao por falso negativo e calculada como a proporcao de falsos negativos em relacao ao total de casos positivos.

## 5. Algoritmo genetico

### Representacao dos genes

Cada individuo representa um conjunto de hiperparametros do XGBoost:

- `max_depth`
- `learning_rate`
- `n_estimators`
- `subsample`
- `colsample_bytree`
- `min_child_weight`
- `gamma`
- `reg_alpha`
- `reg_lambda`

### Populacao inicial

A populacao inicial e gerada aleatoriamente dentro dos limites definidos em `src/tc_fase2/config.py`.

### Selecao

A selecao usa torneio, escolhendo o melhor individuo entre candidatos amostrados da populacao avaliada.

### Crossover

O crossover e uniforme: cada gene do filho vem de um dos dois pais.

### Mutacao

A mutacao sorteia novos valores para genes conforme a taxa definida no experimento.

### Elitismo

O melhor individuo de cada geracao e preservado para a proxima geracao.

### Funcao fitness

A funcao fitness prioriza recall e F1-score e penaliza falsos negativos. Esta escolha esta alinhada ao objetivo clinico de reduzir casos positivos nao detectados.

## 6. Experimentos realizados

Foram implementados tres experimentos:

| Experimento | Populacao | Geracoes | Mutacao |
| --- | ---: | ---: | ---: |
| 1 | 10 | 5 | 0.10 |
| 2 | 20 | 8 | 0.20 |
| 3 | 30 | 10 | 0.30 |

Resultados: preencher apos executar `python -m src.tc_fase2.run_ga_experiments`.

## 7. Resultados

Campos a preencher apos execucao real:

- melhor fitness;
- melhores hiperparametros;
- accuracy;
- recall;
- precision;
- F1-score;
- falsos negativos;
- falsos positivos;
- tempo de execucao.

Arquivos esperados:

- `reports/ga_experiments_summary.csv`
- `reports/ga_fitness_history.csv`

## 8. Comparacao entre modelo original e otimizado

A comparacao e gerada por `python -m src.tc_fase2.compare_models`.

Arquivo esperado:

- `reports/model_comparison.md`

Na analise, recall e falsos negativos devem ter prioridade sobre accuracy isolada.

## 9. Integracao com LLM

O modulo `src/tc_fase2/llm_explainer.py` gera explicacoes em linguagem natural a partir da probabilidade prevista, classe prevista, variaveis clinicas e fatores de influencia fornecidos.

A LLM nao faz diagnostico. Ela apenas explica a saida do modelo.

## 10. Prompt engineering

O prompt instrui a LLM a:

- explicar em portugues claro;
- usar apenas dados fornecidos;
- nao inventar informacoes;
- nao afirmar diagnostico definitivo;
- reforcar que e apoio a decisao clinica;
- destacar fatores clinicos relevantes;
- recomendar avaliacao medica em caso de risco alto.

Arquivo esperado:

- `reports/llm_prompt_used.md`

## 11. Avaliacao das explicacoes geradas

A avaliacao inicial deve verificar se a explicacao:

- menciona a probabilidade prevista;
- diferencia risco elevado e sem risco elevado;
- nao inventa dados ausentes;
- inclui aviso de seguranca medica;
- usa linguagem clara.

## 12. Logging e monitoramento

Os scripts registram inicio, fim, configuracao, melhor fitness por geracao, metricas relevantes, tempo de execucao e erros relevantes.

Arquivos:

- `logs/ga_experiments.log`
- `logs/training.log`

## 13. Testes automatizados

Os testes foram criados em `tests/` e podem ser executados com:

```bash
pytest
```

Eles cobrem componentes do algoritmo genetico, metricas e explicador LLM.

## 14. Desafios encontrados

O projeto original concentrava o treinamento no notebook e a API em um unico arquivo. A Fase 2 foi implementada em um novo pacote para preservar o funcionamento original e organizar os scripts reprodutiveis.

Outro desafio e o custo computacional dos experimentos com XGBoost em bases grandes. Por isso, foi incluido modo `--quick`.

## 15. Conclusao

A Fase 2 adiciona uma esteira reprodutivel para avaliar baseline, otimizar hiperparametros por Algoritmo Genetico, treinar o modelo final, comparar resultados e gerar explicacoes seguras com LLM ou fallback local. Os resultados finais devem ser preenchidos apos a execucao dos experimentos completos.
