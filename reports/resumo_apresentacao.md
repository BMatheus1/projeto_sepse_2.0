# Resumo para apresentação

## Objetivo

Evoluir o modelo de detecção de sepse da Fase 1 com Algoritmo Genético e explicações em linguagem natural.

## O que foi feito

- Avaliação do baseline.
- Otimização de hiperparâmetros do XGBoost com Algoritmo Genético.
- Execução de três experimentos completos.
- Ajuste de threshold em validação.
- Comparação entre baseline e modelo otimizado.
- Geração de explicações com LLM ou fallback local.

## Resultado principal

O modelo otimizado aumentou o recall e reduziu falsos negativos, tornando a triagem mais sensível.

## Trade-off

O ganho de sensibilidade aumentou os falsos positivos e reduziu precision/F1-score.

## Conclusão

O modelo otimizado é mais adequado como triagem sensível, mas não deve ser interpretado como diagnóstico definitivo.
