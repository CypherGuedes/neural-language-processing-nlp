from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
module_path = ROOT / "01-tokenizacao-bpe" / "bpe_tokenizer.py"
spec = spec_from_file_location("bpe_tokenizer", module_path)
module = module_from_spec(spec)
spec.loader.exec_module(module)
BPETokenizer = module.BPETokenizer


def test_roundtrip():
    text = "banana bandana banana"
    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=270)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_vocab_size_limit():
    tokenizer = BPETokenizer()
    try:
        tokenizer.train("abc", vocab_size=100)
    except ValueError:
        pass
    else:
        raise AssertionError("Era esperado ValueError para vocab_size < 256")
