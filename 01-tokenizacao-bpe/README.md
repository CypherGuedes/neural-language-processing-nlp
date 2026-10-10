# 01 - Tokenização BPE

Implementação didática do algoritmo **Byte Pair Encoding (BPE)** em nível de bytes.

O fluxo começa com os 256 valores possíveis de um byte e aprende novas unidades a partir dos pares adjacentes mais frequentes. Cada merge cria um novo token e amplia o vocabulário.

## Conteúdos praticados

- codificação UTF-8 em bytes;
- contagem de pares de tokens;
- seleção do par mais frequente;
- criação iterativa de merges;
- codificação e decodificação;
- análise simples de compressão.

## Arquivos

- `bpe_tokenizer.py`: implementação reutilizável do tokenizador;
- `aplicacao-bpe.ipynb`: notebook demonstrativo, com suporte a uma pasta de arquivos JSON e fallback para um texto de exemplo.

## Execução

Abra o notebook e execute as células em sequência. Caso queira usar um corpus próprio, crie `dados/corpus_bpe/` na raiz do repositório e adicione arquivos `.json` contendo uma chave `text`, individualmente ou em listas de objetos.
