from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module_path = ROOT / "02-modelo-bigrama" / "bigram_model.py"
spec = spec_from_file_location("bigram_model", module_path)
module = module_from_spec(spec)
spec.loader.exec_module(module)
BigramModel = module.BigramModel


class TinyTokenizer:
    eot_token = 0

    def encode(self, text):
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if i != self.eot_token)


def test_bigram_pipeline():
    model = BigramModel(TinyTokenizer(), seed=42)
    train, test = model.prepare_data(["ab.", "ab!", "ac.", "ac!", "ad."])
    model.train(train)
    assert model.calculate_perplexity(test) > 0
    assert isinstance(model.generate_text(max_tokens=10), str)
