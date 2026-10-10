# 02 - Modelo de linguagem por bigramas

Modelo estatístico simples que estima a probabilidade do próximo token a partir do token atual.

## Conteúdos praticados

- tokenização com `tiktoken`;
- separação treino/teste por sentenças;
- contagem de bigramas;
- suavização aditiva;
- cálculo de perplexidade;
- geração probabilística de tokens.

A versão reorganizada preserva a ordem dos tokens dentro de cada sentença. No código original, os tokens eram embaralhados antes da contagem dos bigramas, o que destruía as relações sequenciais que o modelo deveria aprender.

## Arquivos

- `bigram_model.py`: classe reutilizável do modelo;
- `aplicacao-bigrama.ipynb`: treinamento, perplexidade e geração de exemplo.
