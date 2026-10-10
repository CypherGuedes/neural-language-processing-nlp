from __future__ import annotations

from collections import Counter
from typing import Iterable


class BPETokenizer:
    """Tokenizador Byte Pair Encoding em nível de bytes."""

    BASE_VOCAB_SIZE = 256

    def __init__(self) -> None:
        self.vocab: dict[int, bytes] = {
            idx: bytes([idx]) for idx in range(self.BASE_VOCAB_SIZE)
        }
        self.merges: dict[tuple[int, int], int] = {}

    @staticmethod
    def get_stats(ids: list[int]) -> Counter[tuple[int, int]]:
        """Conta pares de tokens adjacentes."""
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Substitui ocorrências não sobrepostas de ``pair`` por ``new_id``."""
        merged: list[int] = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                merged.append(new_id)
                i += 2
            else:
                merged.append(ids[i])
                i += 1
        return merged

    def train(self, text: str, vocab_size: int = 276) -> None:
        """Aprende merges BPE a partir de ``text``."""
        if not text:
            raise ValueError("O texto de treinamento não pode ser vazio.")
        if vocab_size < self.BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size deve ser >= {self.BASE_VOCAB_SIZE}."
            )

        ids = list(text.encode("utf-8"))
        self.vocab = {
            idx: bytes([idx]) for idx in range(self.BASE_VOCAB_SIZE)
        }
        self.merges = {}

        num_merges = vocab_size - self.BASE_VOCAB_SIZE
        for i in range(num_merges):
            stats = self.get_stats(ids)
            if not stats:
                break

            pair, _ = stats.most_common(1)[0]
            new_id = self.BASE_VOCAB_SIZE + i
            ids = self.merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> list[int]:
        """Converte texto em IDs utilizando os merges aprendidos."""
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = self.get_stats(ids)
            if not stats:
                break

            pair = min(
                stats,
                key=lambda candidate: self.merges.get(candidate, float("inf")),
            )
            if pair not in self.merges:
                break

            ids = self.merge(ids, pair, self.merges[pair])

        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Converte IDs para texto UTF-8."""
        try:
            data = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as exc:
            raise ValueError(f"Token desconhecido: {exc.args[0]}") from exc
        return data.decode("utf-8", errors="replace")

    def compression_ratio(self, text: str) -> float:
        """Retorna bytes originais / tokens após BPE."""
        original = len(text.encode("utf-8"))
        encoded = len(self.encode(text))
        return original / encoded if encoded else 0.0


# Compatibilidade com o nome usado na versão original do projeto.
BPE_Tokenizer = BPETokenizer
