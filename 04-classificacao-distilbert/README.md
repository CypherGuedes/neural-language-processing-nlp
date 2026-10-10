# 04 - Classificação de textos com DistilBERT

Fine-tuning de `distilbert-base-uncased` em dois conjuntos de classificação textual.

## Metodologia

- divisão estratificada em 70% treino, 10% validação e 20% teste;
- tokenização com `AutoTokenizer`;
- fine-tuning com `AutoModelForSequenceClassification` e `Trainer`;
- métricas de acurácia, F1 macro e F1 micro;
- matriz de confusão no conjunto de teste.

## Notebooks

- `bert-classic4.ipynb`;
- `bert-industry-sector.ipynb`.

Os notebooks usam caminhos relativos ao repositório e não publicam modelos automaticamente no Hugging Face Hub.

> Observação: treinamento de Transformers pode ser lento em CPU. Para reprodução mais prática, use GPU em Google Colab, Kaggle ou ambiente local compatível.
