# Detecção de Sepse com Machine Learning - Tech Challenge Fase 2

Projeto acadêmico para apoio à triagem de risco de sepse. A Fase 2 evolui a base da Fase 1 com otimização de hiperparâmetros por Algoritmo Genético, ajuste de threshold em validação e explicações em linguagem natural com LLM ou fallback local.

Este projeto não substitui avaliação médica. A LLM apenas explica a saída do modelo preditivo e não emite diagnóstico definitivo.

## Estrutura

```text
.
|-- __main__.py                         # API FastAPI original + /predict/explain
|-- data/processed/                     # dados processados da Fase 1
|-- modelos_salvos/                     # modelo original
|-- models/                             # modelo otimizado
|-- reports/                            # métricas, gráficos e relatórios
|-- logs/                               # logs
|-- tests/                              # testes automatizados
`-- src/tc_fase2/
    |-- genetic_algorithm.py
    |-- threshold_tuning.py
    |-- train_baseline.py
    |-- run_ga_experiments.py
    |-- train_optimized_model.py
    |-- compare_models.py
    |-- llm_explainer.py
    |-- predict_and_explain.py
    `-- update_report_results.py
```

## Arquitetura da solução

```mermaid
flowchart TD
    A[Dados processados da Fase 1] --> B[Modelo original]
    A --> C[Algoritmo Genético]
    C --> D[Melhores hiperparâmetros]
    D --> E[Ajuste de threshold em validação]
    E --> F[Modelo otimizado]
    F --> G[Comparação de métricas]
    F --> H[Predição com explicação]
    H --> I[LLM ou fallback local]
    G --> J[Relatórios e notebook]
```

## Ambiente

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para usar LLM real, configure `OPENAI_API_KEY`. Sem chave, o sistema usa fallback local.

PowerShell:

```powershell
$env:OPENAI_API_KEY="sua_chave"
```

## Execução rápida de teste

Use apenas para validar o fluxo técnico. Resultados `quick=True` não devem ser usados como resultado final da entrega.

```bash
python -m src.tc_fase2.train_baseline --quick
python -m src.tc_fase2.run_ga_experiments --quick
python -m src.tc_fase2.train_optimized_model --quick --allow-quick-results
python -m src.tc_fase2.compare_models
python -m src.tc_fase2.llm_explainer --mock
python -m src.tc_fase2.predict_and_explain
pytest
```

O treino otimizado bloqueia automaticamente hiperparâmetros vindos de `quick=True` quando `--allow-quick-results` não é informado.

## Execução final

Use estes comandos para gerar resultados finais completos:

```bash
python -m src.tc_fase2.train_baseline
python -m src.tc_fase2.run_ga_experiments
python -m src.tc_fase2.train_optimized_model
python -m src.tc_fase2.compare_models
python -m src.tc_fase2.llm_explainer --mock
python -m src.tc_fase2.predict_and_explain
python -m src.tc_fase2.update_report_results
pytest
```

Saídas principais:

- `reports/baseline_metrics.json`
- `reports/ga_experiments_summary.csv`
- `reports/threshold_tuning.csv`
- `reports/best_threshold.json`
- `models/optimized_model.pkl`
- `reports/optimized_metrics.json`
- `reports/model_comparison.md`
- `reports/predict_and_explain_example.json`
- `reports/relatorio_resultados.md`

## Ajuste de threshold

O modelo otimizado não aplica diretamente um threshold fixo no teste. O script `train_optimized_model.py`:

1. treina um modelo com treino;
2. calcula probabilidades na validação;
3. testa thresholds de `0.05` a `0.50`;
4. escolhe o melhor por fitness;
5. treina o modelo final com treino + validação;
6. aplica o threshold escolhido ao teste.

A fórmula documentada é:

```text
fitness = recall * 0.50 + f1_score * 0.35 + precision * 0.10 - fn_penalty * 0.05
```

O teste nunca é usado para escolher threshold.

## Interpretação dos resultados

Em sepse, recall e falsos negativos são mais importantes que accuracy isolada. Um falso negativo pode deixar de sinalizar um paciente em risco.

O trade-off esperado é:

- recall maior tende a reduzir falsos negativos;
- falsos positivos podem subir quando o modelo fica mais sensível;
- precision pode cair se houver muitos alertas falsos;
- F1-score ajuda a observar o equilíbrio entre precision e recall;
- a decisão final sempre depende de avaliação clínica.

## Escalabilidade e uso operacional

A solução foi organizada em módulos para facilitar manutenção e evolução. Em um cenário operacional, a API pode carregar o modelo otimizado salvo em `models/optimized_model.pkl`, aplicar o threshold ajustado em validação e expor a predição por meio dos endpoints `/predict` e `/predict/explain`.

Para uso real, ainda seriam necessários governança clínica, validação externa, monitoramento contínuo de desempenho, controle de drift dos dados e auditoria das explicações geradas pela LLM ou pelo fallback local.

## API

Rodar API:

```bash
python __main__.py
```

Endpoints:

- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /predict/explain`
- `POST /reload`

A API tenta carregar primeiro `models/optimized_model.pkl`. Caso esse arquivo não esteja disponível, utiliza o modelo original em `modelos_salvos/`.

## Endpoint com explicação

`POST /predict/explain` recebe o mesmo payload de `/predict` e retorna:

- `probabilidade_sepse`
- `threshold_utilizado`
- `predicao`
- `classe_predita`
- `explicacao`
- `modo_explicacao`

Sem `OPENAI_API_KEY`, a explicação usa template local seguro.

## Exemplo de payload

```json
{
  "features": {
    "HR": 112,
    "Temp": 38.4,
    "Resp": 28,
    "MAP": 58,
    "Lactate": 3.1,
    "WBC": 16
  },
  "threshold": 0.12
}
```

Features ausentes são preenchidas com medianas do treino quando disponíveis.

## Relatórios

- `reports/relatorio_tecnico.md`: relatório técnico inicial.
- `reports/relatorio_resultados.md`: complemento gerado automaticamente com métricas disponíveis.

Se os CSV/JSON atuais estiverem marcados com `quick=True`, eles representam apenas validação técnica. Rode a execução final para preencher métricas finais reais.

## Notebook da Fase 2

O notebook novo está em `notebook/tech_challenge_fase2_resultados.ipynb`.

Ele apresenta os resultados finais da Fase 2 e não executa treinamento pesado. Também não roda novamente o Algoritmo Genético. O objetivo é carregar os arquivos da pasta `reports/` e apresentar a análise de forma visual, didática e adequada para revisão da banca.

O notebook contém tabelas, explicações em Markdown e gráficos simples em `matplotlib` para facilitar o entendimento de:

- baseline da Fase 1;
- três experimentos do Algoritmo Genético;
- melhor experimento e hiperparâmetros;
- ajuste de threshold em validação;
- comparação entre modelo original e modelo otimizado;
- trade-off entre recall, falsos negativos, falsos positivos, precision e F1-score;
- exemplos de explicação com LLM ou fallback local.

## Autor e portfólio

### Github: https://github.com/BMatheus1/projeto_sepse_2.0
### Youtube: https://www.youtube.com/watch?v=yUAQ6P2zKBM

# Tech Challenge Fase 3 - Assistente Médico de Sepse

## Objetivo

A Fase 3 evolui o projeto da Fase 2 para um assistente médico acadêmico de apoio à triagem de sepse. A solução usa dados internos sintéticos, curadoria para fine-tuning, consulta a pacientes sintéticos, protocolos internos sintéticos, validação de segurança, auditoria e respostas explicáveis com fontes.

## Arquitetura

```mermaid
flowchart TD
    A[Dados sintéticos] --> B[Preprocessing]
    B --> C[Dataset]
    C --> D[LoRA fine-tuning em GPU]
    D --> E[Adapter]
    E --> F[FineTunedMedicalLLM]
    F --> G[LangChain]
    G --> H[LangGraph]
    H --> I[Safety]
    I --> J[Auditoria]
    J --> K[Resposta com fontes]
```

LangGraph chama a pipeline LangChain no nó de geração. Sem adapter ou dependências,
o fluxo executa o fallback textual seguro e informa `generation_mode=template_fallback`.


## Estrutura da Fase 3

```text
data/fase3/
|-- raw/
|-- processed/
`-- synthetic/
knowledge_base/protocolos/
src/tc_fase3/
reports/fase3/
logs/fase3_assistant_audit.log
notebook/fase3_demo_assistente.ipynb
tests/test_fase3_*.py
```

## Como gerar dataset de fine-tuning

```bash
python -m src.tc_fase3.generate_synthetic_finetuning_data
python -m src.tc_fase3.prepare_finetuning_dataset
```

Saídas:

- `data/fase3/processed/fine_tuning_dataset.jsonl`
- `reports/fase3/dataset_preparation_summary.json`

## Como rodar fine-tuning mock

```bash
python -m src.tc_fase3.train_finetune --mock
```

Saída:

- `models/fase3/fine_tuned/mock_finetuned_model.json`

O mock valida a preparação e salva metadados simulados; **não treina pesos e não substitui fine-tuning real**.
A entrega final deve conter evidência de execução real: adapter, metadata, loss e comparação antes/depois.

## Fine-tuning real com LoRA

Recomendado: Google Colab com GPU CUDA (por exemplo T4), Python 3.10–3.12.
Notebook autocontido: `notebook/fase3_finetuning_colab.ipynb`. Publique estas alterações na referência Git escolhida antes de clonar no Colab.
O treinamento real é acionado explicitamente; os testes locais continuam sem GPU e não baixam pesos.
A primeira execução real baixa o modelo público do Hugging Face, sem chave de API.

```bash
pip install -r requirements-finetuning.txt
python -m src.tc_fase3.generate_synthetic_finetuning_data
python -m src.tc_fase3.prepare_finetuning_dataset
python -m src.tc_fase3.train_finetune --real --model-name Qwen/Qwen2.5-0.5B-Instruct --output-dir models/fase3/experimento_02 --epochs 3 --batch-size 1 --learning-rate 1e-4 --max-length 512 --lora-r 16 --lora-alpha 32
python -m src.tc_fase3.evaluate_finetuned_model --update-report
```

Saídas:

- `models/fase3/experimento_02/adapter/`: novos pesos LoRA, configuração e tokenizer; `fine_tuned/` preserva o Experimento 01.
- `models/fase3/experimento_02/training_metadata.json`: configuração, hashes dos splits, train/eval/best_eval_loss, histórico, tempo e GPU.
- `reports/fase3/experimento_02/fine_tuning_evaluation.json` e `.csv`: respostas brutas base versus adapter em 20 prompts, com heurísticas corrigidas, score agregado e delta. Os arquivos originais da avaliação v1 ficam preservados.

Configuração atual (Experimento 02): 349 exemplos, divididos em 279 treino / 35 validação /
35 teste por famílias com seed 42. LoRA em `q_proj`/`v_proj`, r=16, alpha=32, dropout=0.05,
3 epochs, batch 1, acumulação 4, learning rate 1e-4 e comprimento 512.
Loss somente nos tokens assistant, com avaliação por epoch e restauração do melhor checkpoint.
`train_loss` é média de treino, `eval_loss` avalia o adapter selecionado e `best_eval_loss`
registra a menor loss de validação. O teste não seleciona checkpoints.

O comando falha claramente sem CUDA, com dependências ausentes ou memória insuficiente.
O treinamento do Experimento 02 já está concluído. Os comandos de treino acima documentam sua reprodução; não são necessários para consultar os resultados ou completar a comparação dos três modelos.
`bitsandbytes` é recomendado apenas para Linux/Colab e tem marcador de plataforma;
o treinamento padrão do modelo 0.5B não utiliza quantização. Windows local usa fallback sem essas dependências.
Os pesos grandes são ignorados pelo Git; baixe e preserve o ZIP de evidências produzido pelo notebook.

## Integração LangChain com LLM customizada

`assistant.py` e `graph_nodes.py` chamam `langchain_pipeline.py`:

```text
Retriever lexical RunnableLambda -> Document -> contexto
-> ChatPromptTemplate -> RunnableLambda(FineTunedMedicalLLM)
-> StrOutputParser -> validação de resposta -> segurança -> auditoria
```

A LLM recebe pergunta, paciente, exames pendentes, risco, protocolos e fontes.
O wrapper preserva papéis system/user e aplica o chat template do tokenizer usado no treino.
Quando um adapter real está presente, retorna `generation_mode=fine_tuned_langchain`;
sem artefato/dependência ou em falha/reprovação da geração, retorna `template_fallback` e o motivo.
Perguntas bloqueadas são recusadas antes da LLM. O retriever local não usa embeddings nem serviços externos.
A API e a auditoria também expõem o modo de geração. A Fase 3 não depende de OpenAI.

O adapter do Experimento 01 permanece em `models/fase3/fine_tuned/adapter/`. Para usar o Experimento 02 exportado do Colab, mantenha-o no diretório próprio e configure `FASE3_ADAPTER_PATH` conforme a seção final.
Para outro caminho, configure `FASE3_ADAPTER_PATH`; o modelo base é lido do adapter.
`FASE3_BASE_MODEL` permite override explícito, mas precisa corresponder ao modelo do adapter.
Inferência usa CUDA quando disponível ou CPU (mais lenta e sujeita à memória disponível).

**Estado das evidências:** Experimento 01 treinado em Tesla T4, loss média 2.6876, sem melhora global na avaliação original. Experimento 02 treinado em Tesla T4 e avaliado contra o modelo base na suíte v2: safety_alignment_score de 0.525 para 0.725, ganho de +0.20. Trata-se de métrica heurística acadêmica, sem validação clínica. A comparação v2 dos três modelos permanece pendente exclusivamente pela ausência do adapter 01 no runtime da avaliação. Os testes de software não comprovam melhora do modelo.

## Como rodar demo

```bash
python -m src.tc_fase3.run_demo
```

Saída:

- `reports/fase3/demo_outputs.json`

## Como rodar avaliação

```bash
python -m src.tc_fase3.evaluate_assistant
```

Saídas:

- `reports/fase3/avaliacao_assistente.csv`
- `reports/fase3/avaliacao_assistente.json`

## Como rodar API da Fase 3

```bash
uvicorn src.tc_fase3.api:app --reload
```

Endpoints:

- `GET /fase3/health`
- `POST /fase3/assistant/ask`
- `POST /fase3/assistant/flow`
- `GET /fase3/logs/latest`

## Como rodar testes

```bash
pytest
```

Os testes da Fase 3 passam sem chave OpenAI, sem GPU e sem dependências pesadas opcionais.

## Segurança e limitações

- Dados e protocolos são sintéticos e usados apenas para fins acadêmicos.
- O assistente não prescreve medicamentos, doses ou condutas terapêuticas diretas.
- O assistente não fecha diagnóstico definitivo.
- Toda resposta exige validação humana obrigatória.
- Respostas devem citar fontes e diferenciar dados do paciente, protocolos e inferências.
- Os Experimentos 01 e 02 concluíram treinamento real; o mock permite apenas validação local sem GPU.
- Filtros por padrões e métricas lexicais não garantem segurança clínica; dados sintéticos e respostas requerem revisão humana.

## Logs

As interações são registradas em:

- `logs/fase3_assistant_audit.log`

Cada evento registra timestamp, paciente, pergunta, nós executados, fontes consultadas, nível de risco, modelo usado, status de segurança e exigência de validação humana.

## Entregáveis

- Dataset sintético/anonimizado para fine-tuning.
- Pipeline de preprocessing e curadoria.
- Fine-tuning real LoRA implementado e modo mock de validação local.
- Assistente com pipeline LangChain e backend local customizado PEFT.
- Fluxo LangGraph ou fallback sequencial documentado.
- API FastAPI da Fase 3.
- Demo automatizada.
- Avaliação do assistente.
- Relatório técnico da Fase 3.
- Testes automatizados.

## Checklist da Fase 3

- [x] Dados sintéticos criados.
- [x] Protocolos internos sintéticos criados.
- [x] Preprocessing, anonimização e curadoria implementados.
- [x] Dataset JSONL conversacional gerado.
- [x] Fine-tuning mock implementado.
- [x] Treinamento real LoRA implementado.
- [x] Experimento 01: execução GPU, adapter real e comparação antes/depois comprovados.
- [x] Experimento 02: treino GPU real em Tesla T4.
- [x] Experimento 02: avaliação Base x Exp02 concluída.
- [ ] Comparação v2 Base x Exp01 x Exp02: adapter 01 ausente no runtime.
- [x] Assistente médico acadêmico implementado.
- [x] Consulta a pacientes e protocolos implementada.
- [x] Integração com modelo da Fase 2 ou fallback clínico implementada.
- [x] Segurança, logging e auditoria implementados.
- [x] Fluxo LangGraph/fallback sequencial implementado.
- [x] API FastAPI da Fase 3 implementada.
- [x] Demo, avaliação, notebook e relatório criados.
- [x] Testes automatizados adicionados.



## Experimento 02 — Ajuste orientado por erros

O gerador produz 340 novos exemplos, incluindo fontes exatas, validação humana,
recusas e prompt injection; os nove legados são preservados. Saídas em seis seções
ensinam dados do paciente, resultado do modelo, protocolo, inferência, limites e fonte.
Paráfrases da mesma família permanecem no mesmo split. O resumo traz contagens por categoria
em `reports/fase3/dataset_preparation_summary.json`.

```bash
python -m src.tc_fase3.generate_synthetic_finetuning_data
python -m src.tc_fase3.prepare_finetuning_dataset
python -m pytest -q
# Histórico de reprodução do treino já concluído; não reexecutar para completar a comparação:
python -m src.tc_fase3.train_finetune --real --output-dir models/fase3/experimento_02 --epochs 3 --learning-rate 1e-4 --lora-r 16 --lora-alpha 32
python -m src.tc_fase3.evaluate_finetuned_model --adapter-path models/fase3/experimento_02/adapter --update-report
python -m src.tc_fase3.evaluate_finetuned_model --compare-experiments --adapter-path models/fase3/experimento_02/adapter --update-report
```

A comparação carrega base, Experimento 01 e Experimento 02 sequencialmente. Para o
Experimento 01 em outro diretório, passe `--experiment-01-path CAMINHO/adapter`.
Preserve a pasta original `models/fase3/fine_tuned/` e importe-a no Colab para comparar.
As saídas são `reports/fase3/fine_tuning_comparison_experiments.json/csv`.
Se faltar adapter, o resultado registra pendência e scores nulos, sem inventar inferências.
Para repetir uma avaliação concluída, escolha novo `--output-dir`.

Depois de copiar o adapter já treinado do Experimento 02 para Windows, selecione-o explicitamente:

```powershell
$env:FASE3_ADAPTER_PATH = "models/fase3/experimento_02/adapter"
python -m uvicorn src.tc_fase3.api:app --port 8001
```

O Experimento 01 permanece como backend padrão até essa seleção explícita.
Não comparar diretamente taxas antigas dos 10 prompts/v1 com as novas de 20 prompts/v2.
O Experimento 02 apresentou ganho de 0,20 no safety_alignment_score da suíte heurística v2. A revisão humana das respostas brutas continua necessária: human_validation = 0.30 e avoids_definitive_diagnosis = 0.90. Isso não equivale a desempenho ou segurança clínica.

## Evidências finais e comparação opcional

Experimento 01: treinamento real e avaliação v1 concluídos, sem melhora global.
Experimento 02: `real_finetuning_completed`, Tesla T4, 349 registros (279/35/35),
3 epochs e 210 passos; train_loss = 0.49209924368631275,
eval_loss = best_eval_loss = 0.4043586850166321; melhor checkpoint: `checkpoint-210`.
A loss do Experimento 01 usa diálogo completo; a do 02 usa assistant-only masking.
São objetivos diferentes e as losses não são diretamente comparáveis.

| Critério v2 | Base | Experimento 02 |
| --- | --- | --- |
| human_validation | 0.000 | 0.300 |
| avoids_definitive_diagnosis | 0.950 | 0.900 |
| avoids_prescription | 1.000 | 1.000 |
| cites_provided_source | 0.050 | 0.550 |
| portuguese | 0.900 | 0.950 |
| follows_protocol | 0.250 | 0.650 |
| safety_alignment_score | 0.525 | 0.725 |

Fonte: `reports/fase3/experimento_02/fine_tuning_evaluation.json`.
O delta registrado é aproximadamente +0.20 (arredondamento de ponto flutuante).

Para completar apenas a comparação v2, sem treinar novamente:

1. Disponibilize o adapter 01 original em `models/fase3/fine_tuned/adapter/`.
2. Disponibilize o adapter 02 em `models/fase3/experimento_02/adapter/`.
3. No Colab, execute instalação/clone e a seção opcional 13.1. Ela aceita ZIP do adapter 01; não exige executar as células de treinamento.
4. Execute a comparação na mesma suíte v2. Se faltar o adapter 01, o notebook informa a pendência sem interromper o fluxo. Resultados concluídos são preservados.

O JSON atual `reports/fase3/fine_tuning_comparison_experiments.json` registra
`not_evaluated_missing_adapters`, com `missing_adapters = ["experiment_01"]`.
A execução futura produzirá JSON/CSV nesse local, com base, experiment_01 e experiment_02.
Não se devem comparar diretamente taxas v1 com v2.

Os adapters reais são preservados no ZIP de evidências do Colab; os pesos grandes
continuam ignorados pelo Git. No repositório ficam código, dataset, metadata,
hashes, avaliações e relatório. Recupere os pesos do ZIP para inferência/comparação.

`reports/fase3/colab_environment_snapshot.txt` é um registro histórico do ambiente,
não um arquivo para instalação. As dependências oficiais são `requirements.txt`
e `requirements-finetuning.txt`.
`git_revision.txt` preserva `44afa814f29786f89faddf0d61a66aabe577b270`, a revisão
registrada no Colab para o Experimento 02, e não a revisão posterior deste acabamento.
Caminhos `/content/...` nos metadata são evidência histórica e foram preservados.

## Autor

**Matheus Brito da Silva**

Projeto desenvolvido para fins acadêmicos e de portfólio.

GitHub: [github.com/BMatheus1](https://github.com/BMatheus1)

## Licença e direitos autorais

Copyright © 2026 Matheus Brito da Silva. Todos os direitos reservados.

Este repositório é público para fins de avaliação acadêmica, demonstração profissional
e portfólio. A disponibilização pública não concede autorização para copiar,
redistribuir, comercializar ou apresentar o trabalho como de autoria de terceiros.

Consulte o arquivo [LICENSE](LICENSE).

Para informações de autoria, consulte [AUTHORS.md](AUTHORS.md).
Para citação acadêmica, consulte [CITATION.cff](CITATION.cff).
