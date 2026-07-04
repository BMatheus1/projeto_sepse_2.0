# Comparacao entre modelo original e otimizado

| model | accuracy | recall | precision | f1_score | false_negatives | false_positives |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_original | 0.6618 | 0.5590200445434298 | 0.14392201834862386 | 0.22891016871865025 | 198 | 1493 |
| modelo_otimizado_ga | 0.15333333333333332 | 0.9809885931558935 | 0.09237379162191192 | 0.168848167539267 | 5 | 2535 |

Em contexto medico, accuracy isolada nao e suficiente para avaliar um modelo de triagem de sepse.
A prioridade clinica deste projeto e aumentar recall e reduzir falsos negativos, pois um falso negativo pode classificar um paciente com risco de sepse como sem risco.
O modelo otimizado deve ser interpretado como ferramenta academica de apoio a decisao clinica, nunca como diagnostico definitivo.