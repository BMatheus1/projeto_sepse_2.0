# Complemento de resultados

> Aviso: ha resultados marcados como `quick=True`. Eles servem para validacao tecnica do fluxo e nao devem ser usados como resultado final da entrega.
> Para resultados finais, execute os comandos sem `--quick`.

## Baseline

| metrica | valor |
| --- | --- |
| accuracy | 0.6618 |
| recall | 0.5590200445434298 |
| precision | 0.14392201834862386 |
| f1_score | 0.22891016871865025 |
| false_negatives | 198 |
| false_positives | 1493 |

## Experimentos GA

| experiment | population_size | generations | mutation_rate | quick | execution_time_seconds | best_fitness | best_hyperparameters | accuracy | recall | precision | f1_score | false_negatives | false_positives | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 4 | 2 | 0.1 | True | 17.2934 | 0.6332465277777778 | {'max_depth': 7, 'learning_rate': 0.03115, 'n_estimators': 127, 'subsample': 0.68746, 'colsample_bytree': 0.80214, 'min_child_weight': 1, 'gamma': 2.80623, 'reg_alpha': 3.5801, 'reg_lambda': 5.75994} | 0.2683333333333333 | 1.0 | 0.134975369458128 | 0.2378472222222222 | 0 | 878 | 0.12 |
| 2 | 5 | 2 | 0.2 | True | 20.6808 | 0.6268321668261624 | {'max_depth': 4, 'learning_rate': 0.03115, 'n_estimators': 220, 'subsample': 0.69796, 'colsample_bytree': 0.9512, 'min_child_weight': 1, 'gamma': 2.42995, 'reg_alpha': 3.80301, 'reg_lambda': 1.15204} | 0.2541666666666666 | 0.9927007299270072 | 0.1320388349514563 | 0.2330762639245929 | 1 | 894 | 0.12 |
| 3 | 6 | 2 | 0.3 | True | 19.2719 | 0.6271831852959283 | {'max_depth': 2, 'learning_rate': 0.17576, 'n_estimators': 127, 'subsample': 0.81572, 'colsample_bytree': 0.80214, 'min_child_weight': 5, 'gamma': 4.33242, 'reg_alpha': 3.5801, 'reg_lambda': 5.75994} | 0.2583333333333333 | 0.9927007299270072 | 0.1326829268292683 | 0.2340791738382099 | 1 | 889 | 0.12 |

### Melhor experimento

- Experimento: 1
- Fitness: 0.6332465277777778
- Hiperparametros: `{'max_depth': 7, 'learning_rate': 0.03115, 'n_estimators': 127, 'subsample': 0.68746, 'colsample_bytree': 0.80214, 'min_child_weight': 1, 'gamma': 2.80623, 'reg_alpha': 3.5801, 'reg_lambda': 5.75994}`

## Threshold escolhido

- Best threshold: 0.15
- Estrategia: validation_fitness
- Fonte: validation_set

## Modelo otimizado

| metrica | valor |
| --- | --- |
| accuracy | 0.228 |
| recall | 0.9315589353612167 |
| precision | 0.09634290208415258 |
| f1_score | 0.17462580185317178 |
| false_negatives | 18 |
| false_positives | 2298 |

## Comparacao

# Comparacao entre modelo original e otimizado

| model | accuracy | recall | precision | f1_score | false_negatives | false_positives |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_original | 0.6618 | 0.5590200445434298 | 0.14392201834862386 | 0.22891016871865025 | 198 | 1493 |
| modelo_otimizado_ga | 0.15333333333333332 | 0.9809885931558935 | 0.09237379162191192 | 0.168848167539267 | 5 | 2535 |

Em contexto medico, accuracy isolada nao e suficiente para avaliar um modelo de triagem de sepse.
A prioridade clinica deste projeto e aumentar recall e reduzir falsos negativos, pois um falso negativo pode classificar um paciente com risco de sepse como sem risco.
O modelo otimizado deve ser interpretado como ferramenta academica de apoio a decisao clinica, nunca como diagnostico definitivo.

## Exemplo de explicacao

- Modo: template_fallback
- Classe prevista: 1

O modelo preditivo estimou probabilidade de sepse de 72.0% e classificou o paciente como risco elevado de sepse. Recomenda-se avaliacao medica imediata conforme protocolo clinico. Os principais fatores informados associados a decisao foram: MAP baixa, lactato elevado, frequencia respiratoria aumentada. As variaveis clinicas consideradas na explicacao incluem: MAP, Lactate, Resp. Esta explicacao e apenas apoio a decisao clinica, nao e diagnostico definitivo e nao substitui avaliacao de uma equipe medica.