from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42


def localizar_raiz() -> Path:
    return Path(__file__).resolve().parents[2]


def carregar_dataset(caminho: Path) -> pd.DataFrame:
    df = pd.read_csv(caminho)
    colunas = {"text", "class"}
    if not colunas.issubset(df.columns):
        raise ValueError(f"{caminho.name} deve conter as colunas {sorted(colunas)}.")
    return df.dropna(subset=["text", "class"]).copy()


def configuracoes_modelos():
    return {
        "Naive Bayes": (
            MultinomialNB(),
            {"classifier__alpha": [0.1, 0.5, 1.0]},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["lbfgs"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [None, 20],
            },
        ),
    }


def avaliar_dataset(caminho: Path) -> list[dict]:
    df = carregar_dataset(caminho)
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["class"],
        test_size=0.2,
        stratify=df["class"],
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    resultados = []

    for nome, (classificador, parametros) in configuracoes_modelos().items():
        pipeline = Pipeline(
            [
                ("vectorizer", CountVectorizer()),
                ("tfidf", TfidfTransformer()),
                ("classifier", classificador),
            ]
        )

        busca = GridSearchCV(
            pipeline,
            parametros,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        busca.fit(x_train, y_train)
        y_pred = busca.predict(x_test)

        resultados.append(
            {
                "modelo": nome,
                "dataset": caminho.name,
                "melhor_f1_macro_cv": round(busca.best_score_, 4),
                "melhores_parametros": str(busca.best_params_),
                "acuracia_teste": round(accuracy_score(y_test, y_pred), 4),
                "f1_macro_teste": round(f1_score(y_test, y_pred, average="macro"), 4),
                "f1_micro_teste": round(f1_score(y_test, y_pred, average="micro"), 4),
            }
        )

    return resultados


def executar(dataset: str = "todos") -> pd.DataFrame:
    raiz = localizar_raiz()
    datasets = {
        "classic4": raiz / "dados" / "classic4.csv",
        "industry_sector": raiz / "dados" / "industry_sector.csv",
    }

    selecionados = datasets if dataset == "todos" else {dataset: datasets[dataset]}
    resultados = []
    for caminho in selecionados.values():
        print(f"Processando {caminho.name}...")
        resultados.extend(avaliar_dataset(caminho))

    df_resultados = pd.DataFrame(resultados)
    saida = raiz / "03-classificacao-textos" / "resultados" / "grid_search_results_atualizado.csv"
    saida.parent.mkdir(parents=True, exist_ok=True)
    df_resultados.to_csv(saida, index=False)
    print(f"Resultados salvos em: {saida}")
    return df_resultados


def main() -> None:
    parser = argparse.ArgumentParser(description="Classificação de textos com TF-IDF e Grid Search.")
    parser.add_argument(
        "--dataset",
        choices=["classic4", "industry_sector", "todos"],
        default="todos",
    )
    args = parser.parse_args()
    resultados = executar(args.dataset)
    print(resultados.to_string(index=False))


if __name__ == "__main__":
    main()
