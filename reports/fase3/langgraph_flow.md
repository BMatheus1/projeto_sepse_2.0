# Fluxo LangGraph - Assistente de Sepse Fase 3

```mermaid
flowchart TD
    A[Entrada] --> B[Validação de segurança]
    B --> C[Consulta ao paciente sintético]
    C --> D[Verificação de exames pendentes]
    D --> E[Busca em protocolos internos]
    E --> F[Estimativa de risco com modelo Fase 2 ou fallback]
    F --> G[Geração da resposta]
    G --> H[Validação final de segurança]
    H --> I[Log de auditoria]
    I --> J[Resposta final com fontes]
```

Quando `langgraph` está disponível, a estrutura pode ser compilada como grafo. Em ambiente local sem essa dependência, o projeto usa fallback sequencial com os mesmos nós e a mesma ordem de execução.
