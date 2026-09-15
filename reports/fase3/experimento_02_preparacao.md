# Entrega — preparação do Experimento 02

## Estado

Preparação concluída. Treinamento do Experimento 02: **Pendente de execução em GPU**. Nenhuma melhora, loss ou métrica do novo modelo foi inventada.
Adapter, tokenizer, metadata e avaliação do Experimento 01 foram preservados com verificação SHA-256. Fase 2, API, LangChain, LangGraph, fallback e auditoria continuam preservados.

## Dataset

340 exemplos novos em 68 famílias (cinco paráfrases por família) + 9 legados = **349**.
**279 treino / 35 validação / 35 teste**, seed 42, aproximadamente 80/10/10. Famílias e textos idênticos de user/assistant não cruzam splits. Legados ficam no treino. Validação/teste têm famílias de fontes, validação humana, recusas, prompt injection, triagem, exames e alertas.
Saídas padronizadas: Dados do paciente, Resultado do modelo, Protocolo, Inferência, Limites e Fonte. Fontes exatas são fornecidas no contexto. Os quatro erros relatados ganharam famílias de exemplos específicas.

## Treinamento e avaliação

- Loss assistant-only: tokens system/user e cabeçalhos recebem -100; respostas e fim de turno contribuem. Padding também recebe -100.
- Prefixos do chat template são verificados, inclusive em diálogos com múltiplos turnos. Texto malicioso contendo assistant não muda o masking.
- Treino/validação separados, avaliação e salvamento por epoch, seleção da menor eval_loss, preservação do teste e snapshots dos splits.
- Defaults: Qwen 0.5B, 3 epochs, batch 1, acumulação 4, lr 1e-4, max_length 512, r16, alpha32, dropout 0.05, seed42.
- Avaliador v2 com 20 prompts ausentes literalmente no treino. Corrige negações de revisão humana, afirmações de diagnóstico, correspondência exata de fonte e detecção lexical de português.
- Score agregado = média dos seis critérios. Delta = fine-tuned menos base, incluindo valores negativos. Respostas permanecem raw.
- Comparação base/experiment_01/experiment_02 sequencial, com mesmos prompts e critérios. Taxas v1 originais não são misturadas com v2.

## Validação local

- Gerador e preparação executados: 349 registros válidos, zero duplicados e zero descartados.
- `python -m pytest -q`: **99 passed in 16.94s**, incluindo Fases 2 e 3, sem download de modelos e sem GPU.
- Tokenizer real já disponível no adapter 01: todos os 349 exemplos passaram pelo masking; máximo **441 tokens**, abaixo de 512.
- Collator real validado em CPU: preserva labels e mascara padding com -100; nenhum modelo carregado.
- Todas as 340 respostas-alvo passam nos filtros de segurança locais (isso não comprova segurança clínica).
- Notebook validado com nbformat e análise sintática das células Python; GPU ainda não executada.
- `git diff --check`: sem erros. Hashes das evidências do Experimento 01 inalterados.
- Avaliação 02 e comparação dos três modelos registram pendência, scores/delta nulos, sem carregar pesos na ausência do adapter 02.

## Executar no Colab/GPU

```bash
pip install -r requirements-finetuning.txt
python -m src.tc_fase3.generate_synthetic_finetuning_data
python -m src.tc_fase3.prepare_finetuning_dataset
python -m src.tc_fase3.train_finetune --real --output-dir models/fase3/experimento_02 --epochs 3 --batch-size 1 --gradient-accumulation-steps 4 --learning-rate 1e-4 --max-length 512 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --seed 42
python -m src.tc_fase3.evaluate_finetuned_model --adapter-path models/fase3/experimento_02/adapter --update-report
python -m src.tc_fase3.evaluate_finetuned_model --compare-experiments --adapter-path models/fase3/experimento_02/adapter --update-report
```

Publique a versão atual do código antes de clonar no Colab. O adapter 01 é ignorado pelo Git: importe-o do backup/Drive para comparar os três modelos. Saídas de treino e avaliações concluídas são protegidas contra sobrescrita; para nova execução, escolha outro diretório.
As alterações desta tarefa permanecem locais, sem commit/push. Reinicie a API para carregar os filtros atualizados; o adapter 01 permanece como padrão até configurar FASE3_ADAPTER_PATH para o 02.

## Arquivos modificados (13)

- [README.md](D:/projeto_sepse_2.0/README.md)
- [data/fase3/processed/fine_tuning_dataset.jsonl](D:/projeto_sepse_2.0/data/fase3/processed/fine_tuning_dataset.jsonl)
- [data/fase3/raw/finetuning_templates_sinteticos.jsonl](D:/projeto_sepse_2.0/data/fase3/raw/finetuning_templates_sinteticos.jsonl)
- [notebook/fase3_finetuning_colab.ipynb](D:/projeto_sepse_2.0/notebook/fase3_finetuning_colab.ipynb)
- [reports/fase3/dataset_preparation_summary.json](D:/projeto_sepse_2.0/reports/fase3/dataset_preparation_summary.json)
- [reports/fase3/relatorio_tecnico_fase3.md](D:/projeto_sepse_2.0/reports/fase3/relatorio_tecnico_fase3.md)
- [src/tc_fase3/config.py](D:/projeto_sepse_2.0/src/tc_fase3/config.py)
- [src/tc_fase3/evaluate_finetuned_model.py](D:/projeto_sepse_2.0/src/tc_fase3/evaluate_finetuned_model.py)
- [src/tc_fase3/generate_synthetic_finetuning_data.py](D:/projeto_sepse_2.0/src/tc_fase3/generate_synthetic_finetuning_data.py)
- [src/tc_fase3/prepare_finetuning_dataset.py](D:/projeto_sepse_2.0/src/tc_fase3/prepare_finetuning_dataset.py)
- [src/tc_fase3/safety.py](D:/projeto_sepse_2.0/src/tc_fase3/safety.py)
- [src/tc_fase3/train_finetune.py](D:/projeto_sepse_2.0/src/tc_fase3/train_finetune.py)
- [tests/test_fase3_finetuning_pipeline.py](D:/projeto_sepse_2.0/tests/test_fase3_finetuning_pipeline.py)

## Arquivos criados (13)

- [data/fase3/experimento_01/fine_tuning_dataset.jsonl](D:/projeto_sepse_2.0/data/fase3/experimento_01/fine_tuning_dataset.jsonl)
- [data/fase3/experimento_01/finetuning_templates_sinteticos.jsonl](D:/projeto_sepse_2.0/data/fase3/experimento_01/finetuning_templates_sinteticos.jsonl)
- [data/fase3/processed/fine_tuning_test.jsonl](D:/projeto_sepse_2.0/data/fase3/processed/fine_tuning_test.jsonl)
- [data/fase3/processed/fine_tuning_train.jsonl](D:/projeto_sepse_2.0/data/fase3/processed/fine_tuning_train.jsonl)
- [data/fase3/processed/fine_tuning_validation.jsonl](D:/projeto_sepse_2.0/data/fase3/processed/fine_tuning_validation.jsonl)
- [reports/fase3/experimento_01/dataset_preparation_summary.json](D:/projeto_sepse_2.0/reports/fase3/experimento_01/dataset_preparation_summary.json)
- [reports/fase3/experimento_01/original_evidence_sha256.json](D:/projeto_sepse_2.0/reports/fase3/experimento_01/original_evidence_sha256.json)
- [reports/fase3/experimento_02/fine_tuning_evaluation.csv](D:/projeto_sepse_2.0/reports/fase3/experimento_02/fine_tuning_evaluation.csv)
- [reports/fase3/experimento_02/fine_tuning_evaluation.json](D:/projeto_sepse_2.0/reports/fase3/experimento_02/fine_tuning_evaluation.json)
- [reports/fase3/experimento_02_preparacao.md](D:/projeto_sepse_2.0/reports/fase3/experimento_02_preparacao.md)
- [reports/fase3/fine_tuning_comparison_experiments.csv](D:/projeto_sepse_2.0/reports/fase3/fine_tuning_comparison_experiments.csv)
- [reports/fase3/fine_tuning_comparison_experiments.json](D:/projeto_sepse_2.0/reports/fase3/fine_tuning_comparison_experiments.json)
- [tests/test_fase3_experiment_02.py](D:/projeto_sepse_2.0/tests/test_fase3_experiment_02.py)
