# Model Card — Modelo de Detecção de Sepse

## Nome do modelo

Modelo de apoio à triagem de risco de sepse baseado em Machine Learning.

## Tipo de modelo

Classificação binária supervisionada.

## Objetivo

Identificar registros clínicos com maior probabilidade de sepse a partir de sinais vitais, exames laboratoriais e atributos derivados temporalmente.

## Classe alvo

- `0` = sem sepse
- `1` = com sepse

## Contexto de uso

Este modelo foi desenvolvido para fins acadêmicos e de apoio à análise de risco. Ele pode ser interpretado como uma ferramenta de triagem, especialmente útil em cenários onde a sensibilidade é prioritária.

## Dados de entrada

O modelo pode utilizar variáveis como:

- frequência cardíaca (`HR`)
- temperatura (`Temp`)
- frequência respiratória (`Resp`)
- pressão arterial média (`MAP`)
- lactato (`Lactate`)
- leucócitos (`WBC`)
- creatinina (`Creatinine`)
- variáveis derivadas temporais
- scores clínicos compostos
- indicadores de alteração fisiológica

## Estratégia adotada

O pipeline priorizou alta sensibilidade para reduzir falsos negativos. Isso implica um aumento relevante de falsos positivos, o que é coerente com uma estratégia de triagem mais conservadora.

## Métricas observadas no projeto

As análises mostraram, de forma geral:

- recall elevado
- precision baixa
- especificidade moderada a baixa
- presença de muitos falsos positivos
- comportamento sensível ao threshold de decisão

## Pontos fortes

- boa capacidade de capturar sinais clínicos compatíveis com sepse
- uso de variáveis clínicas coerentes com o problema
- possibilidade de interpretação com análise de importância e SHAP
- adequado para rastreamento inicial de risco

## Limitações

- elevado número de falsos positivos
- possível dependência de variáveis contextuais, como unidade/hospital
- possível influência de padrões de missing
- não substitui julgamento clínico
- não validado para uso assistencial real

## Considerações éticas

Este modelo não deve ser usado como ferramenta única para diagnóstico ou tratamento. Seu uso deve ser sempre acompanhado por avaliação clínica humana e validação institucional adequada.

## Riscos

- alarmes excessivos
- sobrecarga operacional
- interpretação incorreta fora do contexto clínico
- queda de desempenho em ambientes diferentes do conjunto de treino/teste

## Recomendações de uso

- usar como apoio à triagem
- sempre combinar com avaliação clínica
- monitorar métricas por ambiente
- recalibrar thresholds conforme o contexto
- validar antes de qualquer aplicação prática

## Status

Projeto acadêmico / experimental.