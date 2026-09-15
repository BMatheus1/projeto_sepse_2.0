# Implementação e validação local da Fase 3

## Resultado

- Treinamento LoRA real implementado para Qwen/Qwen2.5-0.5B-Instruct, com imports opcionais, verificação CUDA, mensagens de erro, proteção contra sobrescrita, adapter e metadata.
- LangChain real (ChatPromptTemplate, RunnableLambda, StrOutputParser, Document) integrado ao assistente e ao LangGraph, com backend FineTunedMedicalLLM e fallback auditável.
- 129 registros preparados: 120 novos determinísticos e 9 originais preservados.
- Inferência, avaliação de 10 prompts base versus adapter, notebook Colab e atualização de relatório implementados.
- Fase 2 preservada: nenhum módulo, dado ou relatório anterior foi alterado ao final.

## Validação executada em 15/09/2026

| Comando ou verificação | Resultado |
| --- | --- |
| `python -m src.tc_fase3.generate_synthetic_finetuning_data` | 120 registros determinísticos |
| `python -m src.tc_fase3.prepare_finetuning_dataset` | 129 válidos, 0 descartados |
| `python -m src.tc_fase3.train_finetune --mock` | Sucesso; sem treinamento de pesos |
| `python -m src.tc_fase3.run_demo` | Sucesso; template_fallback |
| `python -m src.tc_fase3.evaluate_assistant` | 4 casos; taxas de fontes, segurança, validação humana e bloqueio da pergunta perigosa iguais a 1.0; avaliam somente o fallback |
| `python -m src.tc_fase3.evaluate_finetuned_model --update-report` | not_evaluated_no_real_adapter |
| `python -m pytest -q` | 55 passed in 10.59s |
| `git diff --check` | Sem erros |
| Notebook Colab | nbformat.validate e sintaxe das células Python aprovados; nenhuma célula de GPU executada |

Os testes usam LangChain e LangGraph instalados de verdade, com mocks somente no backend pesado. Também exercitam treino, salvamento, OOM, ausência de CUDA, ausência de dependências e inferência CPU/GPU com dublês; isso não demonstra otimização real nem qualidade clínica.
A auditoria dos testes foi isolada em diretórios temporários. O log versionado preserva o histórico anterior e acrescenta apenas execuções da demo e avaliação local com fallback.

## Pendência real e limitações

Este computador tem Python 3.14 e PyTorch 2.10.0+cpu; torch.cuda.is_available() retornou False.
A chamada real encerrou com código 1 e informou dependências ausentes: datasets, peft e accelerate. Nenhum adapter, loss ou metadata de treinamento real foi gerado.
Na demo/avaliação, LangChain emite aviso da camada de compatibilidade Pydantic V1 com Python 3.14; os comandos e testes concluíram. Para o treino, use Python 3.10–3.12 no Colab/Linux com CUDA.
Não declarar a entrega 100% concluída até executar treino GPU, guardar pesos/metadata, comparar respostas e revisar os resultados. Filtros e heurísticas lexicais não garantem segurança ou validade clínica.

## Comandos para GPU/Colab

```bash
pip install -r requirements-finetuning.txt
python -m src.tc_fase3.generate_synthetic_finetuning_data
python -m src.tc_fase3.prepare_finetuning_dataset
python -m src.tc_fase3.train_finetune --real --model-name Qwen/Qwen2.5-0.5B-Instruct --epochs 1 --batch-size 1 --learning-rate 2e-4 --max-length 512 --lora-r 8 --lora-alpha 16
python -m src.tc_fase3.evaluate_finetuned_model --update-report
```

Use o notebook `notebook/fase3_finetuning_colab.ipynb` para exportar o ZIP de evidências e salve também o notebook executado. Publique a implementação na referência Git escolhida antes de clonar no Colab. As alterações desta tarefa estão locais, sem commit ou push.

## Arquivos criados (12)

- [data/fase3/raw/finetuning_templates_sinteticos.jsonl](D:/projeto_sepse_2.0/data/fase3/raw/finetuning_templates_sinteticos.jsonl)
- [notebook/fase3_finetuning_colab.ipynb](D:/projeto_sepse_2.0/notebook/fase3_finetuning_colab.ipynb)
- [reports/fase3/fine_tuning_evaluation.csv](D:/projeto_sepse_2.0/reports/fase3/fine_tuning_evaluation.csv)
- [reports/fase3/fine_tuning_evaluation.json](D:/projeto_sepse_2.0/reports/fase3/fine_tuning_evaluation.json)
- [reports/fase3/implementacao_validacao_local.md](D:/projeto_sepse_2.0/reports/fase3/implementacao_validacao_local.md)
- [requirements-finetuning.txt](D:/projeto_sepse_2.0/requirements-finetuning.txt)
- [src/tc_fase3/evaluate_finetuned_model.py](D:/projeto_sepse_2.0/src/tc_fase3/evaluate_finetuned_model.py)
- [src/tc_fase3/fine_tuned_llm.py](D:/projeto_sepse_2.0/src/tc_fase3/fine_tuned_llm.py)
- [src/tc_fase3/generate_synthetic_finetuning_data.py](D:/projeto_sepse_2.0/src/tc_fase3/generate_synthetic_finetuning_data.py)
- [src/tc_fase3/langchain_pipeline.py](D:/projeto_sepse_2.0/src/tc_fase3/langchain_pipeline.py)
- [tests/test_fase3_finetuning_pipeline.py](D:/projeto_sepse_2.0/tests/test_fase3_finetuning_pipeline.py)
- [tests/test_fase3_langchain_pipeline.py](D:/projeto_sepse_2.0/tests/test_fase3_langchain_pipeline.py)

## Arquivos modificados (25)

- [.gitignore](D:/projeto_sepse_2.0/.gitignore)
- [README.md](D:/projeto_sepse_2.0/README.md)
- [data/fase3/processed/fine_tuning_dataset.jsonl](D:/projeto_sepse_2.0/data/fase3/processed/fine_tuning_dataset.jsonl)
- [logs/fase3_assistant_audit.log](D:/projeto_sepse_2.0/logs/fase3_assistant_audit.log)
- [models/fase3/fine_tuned/mock_finetuned_model.json](D:/projeto_sepse_2.0/models/fase3/fine_tuned/mock_finetuned_model.json)
- [reports/fase3/avaliacao_assistente.csv](D:/projeto_sepse_2.0/reports/fase3/avaliacao_assistente.csv)
- [reports/fase3/avaliacao_assistente.json](D:/projeto_sepse_2.0/reports/fase3/avaliacao_assistente.json)
- [reports/fase3/dataset_preparation_summary.json](D:/projeto_sepse_2.0/reports/fase3/dataset_preparation_summary.json)
- [reports/fase3/demo_outputs.json](D:/projeto_sepse_2.0/reports/fase3/demo_outputs.json)
- [reports/fase3/langgraph_flow.md](D:/projeto_sepse_2.0/reports/fase3/langgraph_flow.md)
- [reports/fase3/relatorio_tecnico_fase3.md](D:/projeto_sepse_2.0/reports/fase3/relatorio_tecnico_fase3.md)
- [src/tc_fase3/assistant.py](D:/projeto_sepse_2.0/src/tc_fase3/assistant.py)
- [src/tc_fase3/audit_logger.py](D:/projeto_sepse_2.0/src/tc_fase3/audit_logger.py)
- [src/tc_fase3/config.py](D:/projeto_sepse_2.0/src/tc_fase3/config.py)
- [src/tc_fase3/evaluate_assistant.py](D:/projeto_sepse_2.0/src/tc_fase3/evaluate_assistant.py)
- [src/tc_fase3/graph_nodes.py](D:/projeto_sepse_2.0/src/tc_fase3/graph_nodes.py)
- [src/tc_fase3/graph_state.py](D:/projeto_sepse_2.0/src/tc_fase3/graph_state.py)
- [src/tc_fase3/langgraph_flow.py](D:/projeto_sepse_2.0/src/tc_fase3/langgraph_flow.py)
- [src/tc_fase3/prepare_finetuning_dataset.py](D:/projeto_sepse_2.0/src/tc_fase3/prepare_finetuning_dataset.py)
- [src/tc_fase3/prompts.py](D:/projeto_sepse_2.0/src/tc_fase3/prompts.py)
- [src/tc_fase3/protocol_retriever.py](D:/projeto_sepse_2.0/src/tc_fase3/protocol_retriever.py)
- [src/tc_fase3/safety.py](D:/projeto_sepse_2.0/src/tc_fase3/safety.py)
- [src/tc_fase3/schemas.py](D:/projeto_sepse_2.0/src/tc_fase3/schemas.py)
- [src/tc_fase3/train_finetune.py](D:/projeto_sepse_2.0/src/tc_fase3/train_finetune.py)
- [tests/conftest.py](D:/projeto_sepse_2.0/tests/conftest.py)
