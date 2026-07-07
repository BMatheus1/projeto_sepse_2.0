# Complemento de resultados

## Baseline

| métrica | valor |
| --- | --- |
| accuracy | 0.6503349759512858 |
| recall | 0.5480532574207514 |
| precision | 0.13543647522213062 |
| f1_score | 0.21719829245147207 |
| false_negatives | 30482 |
| false_positives | 235961 |

## Experimentos do AG

| experiment | population_size | generations | mutation_rate | quick | execution_time_seconds | best_fitness | best_hyperparameters | accuracy | recall | precision | f1_score | false_negatives | false_positives | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10 | 5 | 0.1 | False | 38369.8118 | 0.6353623176672581 | {'max_depth': 7, 'learning_rate': 0.03115, 'n_estimators': 220, 'subsample': 0.69796, 'colsample_bytree': 0.94772, 'min_child_weight': 2, 'gamma': 3.4907, 'reg_alpha': 2.66373, 'reg_lambda': 1.15204} | 0.2176058329928659 | 0.994956398774452 | 0.145096098325497 | 0.2532590241824693 | 107 | 124368 | 0.12 |
| 2 | 20 | 8 | 0.2 | False | 22851.4065 | 0.6353665540585001 | {'max_depth': 5, 'learning_rate': 0.05188, 'n_estimators': 333, 'subsample': 0.86305, 'colsample_bytree': 0.76662, 'min_child_weight': 1, 'gamma': 1.16312, 'reg_alpha': 0.82961, 'reg_lambda': 5.9541} | 0.2326220182909582 | 0.9931652132924818 | 0.1473313241638755 | 0.2565976154811053 | 145 | 121941 | 0.12 |
| 3 | 30 | 10 | 0.3 | False | 173956.4061 | 0.6357576428258713 | {'max_depth': 6, 'learning_rate': 0.03278, 'n_estimators': 371, 'subsample': 0.69849, 'colsample_bytree': 0.82084, 'min_child_weight': 3, 'gamma': 2.79547, 'reg_alpha': 2.1911, 'reg_lambda': 3.96307} | 0.2351676671171312 | 0.993400895592741 | 0.1477744432602232 | 0.2572773162588277 | 140 | 121541 | 0.12 |

### Melhor experimento

- Experimento: 3
- Fitness: 0.6357576428258713
- Hiperparâmetros: `{'max_depth': 6, 'learning_rate': 0.03278, 'n_estimators': 371, 'subsample': 0.69849, 'colsample_bytree': 0.82084, 'min_child_weight': 3, 'gamma': 2.79547, 'reg_alpha': 2.1911, 'reg_lambda': 3.96307}`

## Threshold escolhido

- Best threshold: 0.15
- Estratégia: validation_fitness
- Fonte: validation_set

## Modelo otimizado

| métrica | valor |
| --- | --- |
| accuracy | 0.24004225749512792 |
| recall | 0.9251697654419833 |
| precision | 0.09804442237711254 |
| f1_score | 0.17729960419502133 |
| false_negatives | 5047 |
| false_positives | 574037 |

## Comparação

# Comparação entre modelo original e otimizado

## Métricas

| model | accuracy | recall | precision | f1_score | false_negatives | false_positives |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_original | 0.6503349759512858 | 0.5480532574207514 | 0.13543647522213062 | 0.21719829245147207 | 30482 | 235961 |
| modelo_otimizado_ga | 0.24004225749512792 | 0.9251697654419833 | 0.09804442237711254 | 0.17729960419502133 | 5047 | 574037 |

## Diferenças absolutas

| metric | baseline | optimized | absolute_difference |
| --- | --- | --- | --- |
| recall | 0.5480532574207514 | 0.9251697654419833 | 0.37711650802123187 |
| precision | 0.13543647522213062 | 0.09804442237711254 | -0.03739205284501808 |
| f1_score | 0.21719829245147207 | 0.17729960419502133 | -0.03989868825645074 |
| false_negatives | 30482.0 | 5047.0 | -25435.0 |
| false_positives | 235961.0 | 574037.0 | 338076.0 |

## Análise automática

- Falsos negativos caíram em 25435, indicando ganho de sensibilidade.
- Falsos positivos subiram em 338076, gerando mais alertas falsos e possível custo operacional.
- Precision caiu; isso evidencia o trade-off de ampliar sensibilidade aceitando mais alertas falsos.
- Em sepse, recall e redução de falsos negativos são mais críticos do que accuracy isolada.
- Este modelo é acadêmico e deve ser usado apenas como apoio, nunca como diagnóstico definitivo.

Em contexto médico, accuracy isolada não é suficiente para avaliar um modelo de triagem de sepse.
A prioridade clínica deste projeto é aumentar recall e reduzir falsos negativos, pois um falso negativo pode classificar um paciente com risco de sepse como sem risco.
O modelo otimizado deve ser interpretado como ferramenta acadêmica de apoio à decisão clínica, nunca como diagnóstico definitivo.

## Exemplo de explicação

- Modo: template_fallback
- Classe prevista: 1

O modelo preditivo estimou probabilidade de sepse de 72.0% e classificou o paciente como risco elevado de sepse. Recomenda-se avaliação médica imediata conforme protocolo clínico. Os principais fatores informados associados à decisão foram: MAP baixa, lactato elevado, frequência respiratória aumentada. As variáveis clínicas consideradas na explicação incluem: MAP, Lactate, Resp. Esta explicação é apenas apoio à decisão clínica, não é diagnóstico definitivo e não substitui avaliação de uma equipe médica.
