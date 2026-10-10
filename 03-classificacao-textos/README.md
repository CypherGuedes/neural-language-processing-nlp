# 03 - Classificação de textos com Machine Learning

Comparação de três classificadores supervisionados aplicados a representações TF-IDF:

- Multinomial Naive Bayes;
- Logistic Regression;
- Random Forest.

A busca de hiperparâmetros usa `GridSearchCV` com validação cruzada estratificada e a seleção é feita por F1 macro. Após a escolha dos melhores parâmetros, o modelo é avaliado no conjunto de teste com acurácia, F1 macro e F1 micro.

## Execução

Na raiz do repositório:

```bash
python 03-classificacao-textos/src/classificacao.py --dataset todos
```

O script utiliza os datasets presentes em `dados/` e salva um CSV atualizado em `03-classificacao-textos/resultados/grid_search_results_atualizado.csv`.

O arquivo `resultados/grid_search_results.csv` preserva os resultados registrados no experimento original.
