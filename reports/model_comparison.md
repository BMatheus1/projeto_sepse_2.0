# Comparacao entre modelo original e otimizado

## Metricas

| model | accuracy | recall | precision | f1_score | false_negatives | false_positives |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_original | 0.6503349759512858 | 0.5480532574207514 | 0.13543647522213062 | 0.21719829245147207 | 30482 | 235961 |
| modelo_otimizado_ga | 0.24004225749512792 | 0.9251697654419833 | 0.09804442237711254 | 0.17729960419502133 | 5047 | 574037 |

## Diferencas absolutas

| metric | baseline | optimized | absolute_difference |
| --- | --- | --- | --- |
| recall | 0.5480532574207514 | 0.9251697654419833 | 0.37711650802123187 |
| precision | 0.13543647522213062 | 0.09804442237711254 | -0.03739205284501808 |
| f1_score | 0.21719829245147207 | 0.17729960419502133 | -0.03989868825645074 |
| false_negatives | 30482.0 | 5047.0 | -25435.0 |
| false_positives | 235961.0 | 574037.0 | 338076.0 |

## Analise automatica

- Falsos negativos cairam em 25435, indicando ganho de sensibilidade.
- Falsos positivos subiram em 338076, gerando mais alertas falsos e possivel custo operacional.
- Precision caiu; isso evidencia o trade-off de ampliar sensibilidade aceitando mais alertas falsos.
- Em sepse, recall e reducao de falsos negativos sao mais criticos do que accuracy isolada.
- Este modelo e academico e deve ser usado apenas como apoio, nunca como diagnostico definitivo.

Em contexto medico, accuracy isolada nao e suficiente para avaliar um modelo de triagem de sepse.
A prioridade clinica deste projeto e aumentar recall e reduzir falsos negativos, pois um falso negativo pode classificar um paciente com risco de sepse como sem risco.
O modelo otimizado deve ser interpretado como ferramenta academica de apoio a decisao clinica, nunca como diagnostico definitivo.