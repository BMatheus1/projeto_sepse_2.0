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