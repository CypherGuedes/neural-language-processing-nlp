# Processamento de Linguagem Natural com Python

Repositório acadêmico com implementações e experimentos de **Processamento de Linguagem Natural (PLN)**, organizados em uma sequência que parte de fundamentos de tokenização e modelos estatísticos até classificação com Transformers e geração de texto com GPT-2.

O projeto reúne cinco estudos práticos: implementação de um tokenizador **Byte Pair Encoding (BPE)**, modelo de linguagem por **bigramas**, classificação de textos com algoritmos clássicos de aprendizado de máquina, classificação com **DistilBERT** e geração de texto com **GPT-2**.

## Objetivos

- compreender como textos são convertidos em tokens;
- implementar um tokenizador BPE a partir dos princípios do algoritmo;
- estudar modelos probabilísticos simples de linguagem com bigramas;
- comparar classificadores tradicionais para tarefas de classificação textual;
- aplicar Transformers pré-treinados à classificação de textos;
- carregar pesos pré-treinados do GPT-2 em uma implementação própria e gerar texto;
- organizar experimentos de PLN de forma reproduzível e adequada para portfólio técnico.

## Projetos

| Módulo | Tema | Principais conceitos |
| --- | --- | --- |
| `01-tokenizacao-bpe` | Tokenização BPE | bytes, pares frequentes, merges, vocabulário, encode/decode |
| `02-modelo-bigrama` | Modelo estatístico de linguagem | bigramas, probabilidade condicional, suavização, perplexidade, geração |
| `03-classificacao-textos` | Classificação com ML clássico | Bag of Words, TF-IDF, Naive Bayes, Logistic Regression, Random Forest, Grid Search |
| `04-classificacao-distilbert` | Classificação com Transformer | tokenização, embeddings, fine-tuning, F1 macro/micro, acurácia, matriz de confusão |
| `05-geracao-gpt2` | Geração de texto | Transformer decoder, transferência de pesos, top-k sampling, temperatura |

## Estrutura do repositório

```text
processamento-linguagem-natural/
├── 01-tokenizacao-bpe/
│   ├── README.md
│   ├── aplicacao-bpe.ipynb
│   └── bpe_tokenizer.py
├── 02-modelo-bigrama/
│   ├── README.md
│   ├── aplicacao-bigrama.ipynb
│   └── bigram_model.py
├── 03-classificacao-textos/
│   ├── README.md
│   ├── aplicacao-classificacao.ipynb
│   ├── resultados/
│   │   └── grid_search_results.csv
│   └── src/
│       ├── __init__.py
│       └── classificacao.py
├── 04-classificacao-distilbert/
│   ├── README.md
│   ├── bert-classic4.ipynb
│   └── bert-industry-sector.ipynb
├── 05-geracao-gpt2/
│   ├── README.md
│   └── gpt2.ipynb
├── dados/
│   ├── README.md
│   ├── classic4.csv
│   └── industry_sector.csv
├── .gitignore
├── GUIA-GITHUB.md
└── requirements.txt
```

## Conjuntos de dados

Os experimentos de classificação usam dois conjuntos de dados disponibilizados no contexto da disciplina:

- **classic4**: coleção de documentos distribuídos em quatro classes;
- **industry_sector**: coleção de documentos classificados por setores da indústria.

Os arquivos foram centralizados em `dados/` para remover dependências de caminhos absolutos do computador local. Informações adicionais estão em [`dados/README.md`](dados/README.md).

## Resultados registrados da classificação clássica

O arquivo `03-classificacao-textos/resultados/grid_search_results.csv` mantém o registro de resultados do experimento original. Os valores abaixo são apresentados como histórico do trabalho e podem variar quando o treinamento for executado novamente, dependendo das versões das bibliotecas e da configuração computacional.

| Modelo | Dataset | Melhor score CV | Score de teste |
| --- | --- | ---: | ---: |
| Naive Bayes | classic4 | 0,85 | 0,84 |
| Logistic Regression | classic4 | 0,87 | 0,89 |
| Random Forest | classic4 | 0,86 | 0,87 |
| Naive Bayes | industry_sector | 0,82 | 0,81 |
| Logistic Regression | industry_sector | 0,84 | 0,85 |
| Random Forest | industry_sector | 0,85 | 0,86 |

## Como executar

### 1. Clonar o repositório

```bash
git clone https://github.com/SEU-USUARIO/processamento-linguagem-natural.git
cd processamento-linguagem-natural
```

### 2. Criar um ambiente virtual

No Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

No Linux ou macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Abrir os notebooks

```bash
jupyter lab
```

Para os experimentos com DistilBERT e GPT-2, uma GPU é recomendada. O código também pode ser executado em ambientes como Google Colab ou Kaggle, com os ajustes de caminho necessários.

## Executando a classificação clássica por linha de comando

Para avaliar os dois datasets:

```bash
python 03-classificacao-textos/src/classificacao.py --dataset todos
```

Para executar apenas um conjunto:

```bash
python 03-classificacao-textos/src/classificacao.py --dataset classic4
```

ou:

```bash
python 03-classificacao-textos/src/classificacao.py --dataset industry_sector
```

Os resultados são gravados em `03-classificacao-textos/resultados/`.

## Testes rápidos

```bash
pytest -q
```

Os testes cobrem o ciclo de codificação/decodificação do BPE e o fluxo básico do modelo de bigramas.

## Melhorias aplicadas na organização

Esta versão foi reorganizada para uso como portfólio. Entre os principais ajustes estão:

- remoção de caminhos absolutos ligados a um computador específico;
- padronização dos nomes de pastas e arquivos;
- centralização dos datasets;
- consolidação de scripts repetidos de classificação;
- correção da divisão dos dados do modelo de bigramas para preservar sequências;
- cálculo de perplexidade com tratamento consistente de suavização;
- notebooks BERT com divisão estratificada de 70% treino, 10% validação e 20% teste;
- avaliação com acurácia, F1 macro e F1 micro;
- remoção da publicação automática no Hugging Face durante o treinamento;
- implementação completa da arquitetura GPT usada no experimento e transferência dos pesos do GPT-2 124M;
- correção do top-k sampling para preservar os IDs reais dos tokens;
- inclusão de `.gitignore`, `requirements.txt` e documentação específica por módulo.

## Tecnologias

Python, Jupyter Notebook, NumPy, pandas, scikit-learn, PyTorch, TensorFlow, Hugging Face Transformers, Hugging Face Datasets, tiktoken, NLTK, Matplotlib e UMAP.

## Observações

Este é um repositório de estudo. Alguns experimentos, principalmente DistilBERT e GPT-2, exigem download de modelos ou pesos externos e podem demandar memória e tempo de processamento significativos.

