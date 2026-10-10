from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable


class BigramModel:
    """Modelo de linguagem de bigramas com suavização aditiva."""

    def __init__(self, tokenizer, alpha: float = 1.0, seed: int = 42) -> None:
        if alpha <= 0:
            raise ValueError("alpha deve ser maior que zero.")
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.seed = seed
        self.bigram_counts: dict[int, Counter[int]] = defaultdict(Counter)
        self.context_counts: Counter[int] = Counter()
        self.vocab: set[int] = set()
        self.boundary_token = getattr(tokenizer, "eot_token", None)

    def _encode_sentence(self, sentence: str) -> list[int]:
        ids = list(self.tokenizer.encode(sentence))
        if self.boundary_token is None:
            return ids
        return [self.boundary_token, *ids, self.boundary_token]

    def prepare_data(
        self,
        sentences: Iterable[str],
        train_ratio: float = 0.8,
    ) -> tuple[list[int], list[int]]:
        """Divide por sentenças e só depois concatena os tokens."""
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio deve estar entre 0 e 1.")

        prepared = [s.strip() for s in sentences if s and s.strip()]
        if len(prepared) < 2:
            raise ValueError("Forneça pelo menos duas sentenças.")

        rng = random.Random(self.seed)
        rng.shuffle(prepared)
        split = max(1, min(len(prepared) - 1, int(len(prepared) * train_ratio)))

        train_sentences = prepared[:split]
        test_sentences = prepared[split:]

        train_tokens = [
            token
            for sentence in train_sentences
            for token in self._encode_sentence(sentence)
        ]
        test_tokens = [
            token
            for sentence in test_sentences
            for token in self._encode_sentence(sentence)
        ]
        return train_tokens, test_tokens

    def train(self, train_tokens: list[int]) -> None:
        if len(train_tokens) < 2:
            raise ValueError("São necessários ao menos dois tokens para treinar.")

        self.bigram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set(train_tokens)

        for first, second in zip(train_tokens, train_tokens[1:]):
            self.bigram_counts[first][second] += 1
            self.context_counts[first] += 1

    def transition_probability(self, first: int, second: int) -> float:
        if not self.vocab:
            raise RuntimeError("Treine o modelo antes de calcular probabilidades.")

        vocab_size = len(self.vocab)
        numerator = self.bigram_counts[first][second] + self.alpha
        denominator = self.context_counts[first] + self.alpha * vocab_size
        return numerator / denominator

    def calculate_perplexity(self, tokens: list[int]) -> float:
        if len(tokens) < 2:
            raise ValueError("São necessários ao menos dois tokens para perplexidade.")

        log_prob_sum = 0.0
        n_bigrams = 0
        for first, second in zip(tokens, tokens[1:]):
            probability = self.transition_probability(first, second)
            log_prob_sum += math.log(probability)
            n_bigrams += 1

        return math.exp(-log_prob_sum / n_bigrams)

    def generate_text(self, max_tokens: int = 40) -> str:
        if not self.vocab:
            raise RuntimeError("Treine o modelo antes de gerar texto.")

        rng = random.Random(self.seed)
        current = self.boundary_token
        if current is None or current not in self.bigram_counts:
            current = max(self.context_counts, key=self.context_counts.get)

        generated: list[int] = []
        for _ in range(max_tokens):
            candidates = self.bigram_counts.get(current)
            if not candidates:
                break

            tokens = list(candidates.keys())
            weights = [count + self.alpha for count in candidates.values()]
            next_token = rng.choices(tokens, weights=weights, k=1)[0]

            if self.boundary_token is not None and next_token == self.boundary_token:
                if generated:
                    break
                current = next_token
                continue

            generated.append(next_token)
            current = next_token

        return self.tokenizer.decode(generated)
