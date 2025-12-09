# 🎬 Agente Inteligente de Recomendação de Filmes

> Um sistema híbrido que une ométodo matemático do PCA e Agentes de IA (LangGraph + Gemini) para recomendações personalizadas de cinema.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![AI](https://img.shields.io/badge/AI-Gemini_Flash-orange)
![Math](https://img.shields.io/badge/Math-PCA_Rank_20-green)
![Status](https://img.shields.io/badge/Status-Concluído-brightgreen)

## 📋 Sobre o Projeto

Este projeto foi desenvolvido como parte da disciplina de **Computação Científica e Analise de Dados (COCADA)**. O objetivo foi aplicar conceitos matemáticos teóricos em uma aplicação prática de Inteligência Artificial.

O sistema resolve o problema da **sobrecarga de escolha** em plataformas de streaming, oferecendo:
1.  **Recomendações Matemáticas:** Baseadas no histórico de 100k avaliações (MovieLens).
2.  **Análise de Perfil:** Um agente que interpreta dados brutos para entender o gosto do usuário.
3.  **Contexto Semântico (RAG):** Busca informações sobre filmes (sinopses, diretores) em linguagem natural.

---

## ⚙️ Arquitetura e Funcionalidades

O sistema é orquestrado pelo **LangGraph**, que decide qual ferramenta acionar com base na intenção do usuário:

### 1. Motor de Recomendação (Math Tool)
Implementação manual do algoritmo de **PCA (Principal Component Analysis)**.
* **Método:** Aproximação de Posto K ($k=20$).
* **Cálculo:** Decomposição Espectral da Matriz de Covariância ($C = A^T A$).
  
### 2. Analista de Dados (Profile Tool)
Um agente especializado que lê os dados brutos do usuário (filmes assistidos, notas, gêneros) e utiliza LLM (Large Language Model) para responder perguntas qualitativas, como *"Qual meu gênero favorito?"* ou *"Eu gosto de filmes antigos?"*.

### 3. Memória Semântica (RAG Tool)
Utiliza **Embeddings Locais** (HuggingFace) e **FAISS** para criar um banco de dados vetorial pesquisável. Permite responder perguntas como *"Quem dirigiu Toy Story?"* ou *"Sobre o que é Matrix?"*.

---

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python
* **IA Generativa:** Google Gemini 2.5 Flash
* **Orquestração:** LangChain & LangGraph
* **Matemática:** NumPy & Pandas
* **Banco Vetorial:** FAISS & HuggingFace Embeddings (Local)
