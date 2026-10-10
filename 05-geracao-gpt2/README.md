# 05 - Geração de texto com GPT-2

Implementação didática da arquitetura GPT e transferência dos pesos públicos do GPT-2 124M para o modelo em PyTorch.

## Conteúdos praticados

- embeddings de token e posição;
- atenção causal multi-head;
- blocos Transformer;
- normalização e feed-forward;
- carregamento dos checkpoints originais do GPT-2;
- transferência de pesos para a implementação local;
- geração autoregressiva;
- controle por temperatura e top-k sampling.

A versão original do notebook definia `GPTModel` sem incluir algumas classes necessárias e gerava texto com um erro no top-k: a amostragem usava a posição dentro dos `k` maiores logits como se fosse o ID do vocabulário. Nesta versão, a arquitetura está completa e os IDs reais dos tokens são preservados.

O notebook baixa os pesos públicos do GPT-2 124M na primeira execução. Esses arquivos são ignorados pelo Git para evitar armazenar checkpoints grandes no repositório.
