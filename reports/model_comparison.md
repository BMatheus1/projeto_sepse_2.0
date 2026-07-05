# Comparacao entre modelo original e otimizado

## Metricas

| model | accuracy | recall | precision | f1_score | false_negatives | false_positives |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_original | 0.6618 | 0.5590200445434298 | 0.14392201834862386 | 0.22891016871865025 | 198 | 1493 |
| modelo_otimizado_ga | 0.228 | 0.9315589353612167 | 0.09634290208415258 | 0.17462580185317178 | 18 | 2298 |

## Diferencas absolutas

| metric | baseline | optimized | absolute_difference |
| --- | --- | --- | --- |
| recall | 0.5590200445434298 | 0.9315589353612167 | 0.3725388908177869 |
| precision | 0.14392201834862386 | 0.09634290208415258 | -0.04757911626447128 |
| f1_score | 0.22891016871865025 | 0.17462580185317178 | -0.05428436686547847 |
| false_negatives | 198.0 | 18.0 | -180.0 |
| false_positives | 1493.0 | 2298.0 | 805.0 |

## Analise automatica

- Falsos negativos cairam em 180, indicando ganho de sensibilidade.
- Falsos positivos subiram em 805, gerando mais alertas falsos e possivel custo operacional.
- Precision caiu; isso evidencia o trade-off de ampliar sensibilidade aceitando mais alertas falsos.
- Em sepse, recall e reducao de falsos negativos sao mais criticos do que accuracy isolada.
- Este modelo e academico e deve ser usado apenas como apoio, nunca como diagnostico definitivo.

## Aviso sobre modo quick

Pelo menos um arquivo de metricas foi gerado com `quick=True`. Esses numeros servem para validacao tecnica do fluxo e nao devem ser usados como resultado final da entrega.
Para gerar resultados finais, execute os comandos sem `--quick`.

Em contexto medico, accuracy isolada nao e suficiente para avaliar um modelo de triagem de sepse.
A prioridade clinica deste projeto e aumentar recall e reduzir falsos negativos, pois um falso negativo pode classificar um paciente com risco de sepse como sem risco.
O modelo otimizado deve ser interpretado como ferramenta academica de apoio a decisao clinica, nunca como diagnostico definitivo.