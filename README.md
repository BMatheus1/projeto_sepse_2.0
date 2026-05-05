# 🩺 Detecção de Sepse com Machine Learning

<p align="center">
  Projeto de Machine Learning para apoio à <strong>triagem de risco de sepse</strong>,
  com foco em <strong>alta sensibilidade</strong>, análise de erros e interpretabilidade clínica.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-Modelo-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Acadêmico-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Licença-MIT-black?style=for-the-badge" />
</p>

---

## 📌 Sobre o projeto

Este projeto foi desenvolvido com o objetivo de construir um modelo capaz de identificar pacientes com maior probabilidade de **sepse** a partir de:

- sinais vitais
- exames laboratoriais
- variáveis clínicas
- atributos temporais derivados

A proposta é atuar como uma ferramenta de **apoio à triagem**, priorizando a detecção de casos suspeitos e permitindo análise detalhada de:

- **recall**
- **precision**
- **specificity**
- **falsos positivos**
- **falsos negativos**
- **importância das variáveis**
- **interpretabilidade com SHAP**

---

## 🎯 Objetivo

Desenvolver um pipeline de Machine Learning para apoio à detecção de sepse com foco em:

- **alta sensibilidade**
- redução do risco de perder casos reais
- interpretação clínica dos resultados
- análise aprofundada dos erros do modelo

---

## 🧠 Estratégia adotada

Como a sepse é um problema clínico crítico, a modelagem foi orientada para **priorizar sensibilidade**.

Na prática, isso significa que o projeto busca:

- capturar o maior número possível de casos positivos
- aceitar um volume maior de falsos positivos quando necessário
- analisar se esses falsos positivos são apenas ruído ou também representam pacientes clinicamente instáveis

Além do classificador principal, o projeto também explora a ideia de duas saídas finais:

- **triagem sensível**
- **alerta forte**

Essa abordagem permite comparar um cenário mais sensível com outro mais restritivo.

---

## 🗂️ Estrutura do projeto

```text
.
│   Dockerfile
│   License
│   MODEL_CARD.md
│   README.md
│   __main__.py
│
├───data
│   └───processed
│           features_modelo_com_tempo_admin.csv
│           features_modelo_sem_tempo_admin.csv
│           medianas_treino_com_tempo_admin.csv
│           medianas_treino_sem_tempo_admin.csv
│           test_melhor.parquet
│           train_melhor.parquet
│           val_melhor.parquet
│
├───modelos_salvos
│       modelo_sepse_sem_tempo_admin.pkl
│
└───notebook
    │   analise_sepse_2.0.ipynb
    │
    └───catboost_info
        │   catboost_training.json
        │   learn_error.tsv
        │   time_left.tsv

```
# ▶️ Como executar o projeto

## 1. Executando localmente

### Pré-requisitos

- Python 3.11 ou superior
- pip instalado

### Instalar as dependências

```bash
pip install -r requirements.txt
```
### Rodar a API:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
### Acessar no navegador:

- API: http://localhost:8000
- Documentação interativa: http://localhost:8000/docs
- Status da API: http://localhost:8000/health

## 2. Executando com Docker

### Construir a imagem: 

```bash
docker build -t projeto-sepse .
```
### Rodar o container: 

```bash
docker run -p 8000:8000 projeto-sepse
```
## Acessar no navegador

- API: http://localhost:8000
- Documentação interativa: http://localhost:8000/docs
- Status da API: http://localhost:8000/health

## 3. Testando a API

### Teste simples de saúde: 

curl http://localhost:8000/health

### Exemplo de predição:
```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "features": {
    "HR": 110,
    "Temp": 38.2,
    "Resp": 24,
    "MAP": 62,
    "Lactate": 2.8,
    "WBC": 16.5,
    "Creatinine": 1.9,
    "Age": 67,
    "SBP": 95,
    "DBP": 58,
    "O2Sat": 91,
    "FiO2": 0.4,
    "Platelets": 120,
    "BUN": 34
  },
  "threshold": 0.12
}'
```
## 🩺 Endpoints disponíveis

- GET /

### Retorna uma mensagem indicando que a API está ativa.

- GET /health

### Retorna o status da aplicação e informações sobre carregamento do modelo.

- GET /metadata

### Retorna metadados da API e do modelo carregado.

- POST /predict

### Recebe as variáveis clínicas do paciente e retorna a probabilidade de sepse e a classe predita.

- POST /reload

### Recarrega o artefato salvo sem precisar reiniciar a aplicação.

# 📦 Arquivos necessários

## Para a API funcionar corretamente, estes arquivos devem existir no projeto.

### Modelo salvo:

modelos_salvos/modelo_sepse_sem_tempo_admin.pkl

### Arquivos auxiliares:

data/processed/features_modelo_sem_tempo_admin.csv
data/processed/medianas_treino_sem_tempo_admin.csv

## 📊 Notebook principal

### O treinamento, avaliação, análise e geração do artefato .pkl são feitos no notebook:

- notebooks/analise_sepse_2.0.ipynb

### Esse notebook inclui:

- leitura e preparação dos dados
- engenharia de features
- treinamento dos modelos
- comparação de métricas
- análise de desempenho
- salvamento do modelo final




## 👨‍💻 Grupo: 

Matheus Brito da Silva rm373928
Ricardo Pinto rm374174
Felipe Monay rm366815
Ari Monteiro rm371705
Pedro Artur Araújo Pinto rm373866

```