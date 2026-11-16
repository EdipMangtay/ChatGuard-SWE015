"""Trainable word2vec/MLP and deployment model loaders."""
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from .quantize import quantize_bert

MODEL_NAMES = {
    "tfidf": "TF-IDF + logistic regression",
    "mlp": "Word2vec + feedforward",
    "bert": "BERT Tiny FP32",
    "int8": "BERT Tiny INT8",
}
MAX_LENGTH = 64
VOCAB_LIMIT = 12000
EMBED_DIM = 64


def init_cpu(threads: int = 2) -> str:
    torch.set_num_threads(threads)
    return "PyTorch CPU (W8A32)"

def make_tokenizer(path: Path) -> BertTokenizerFast:
    return BertTokenizerFast(vocab=str(path), do_lower_case=True, model_max_length=MAX_LENGTH)


def encode_words(texts: list[str], vocab: dict[str, int]) -> torch.Tensor:
    tokens = [[vocab.get(word, 1) for word in text.split()[:MAX_LENGTH]] for text in texts]
    return torch.tensor(
        [row + [0] * (MAX_LENGTH - len(row)) for row in tokens], dtype=torch.long
    )


class WordMLP(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(EMBED_DIM, 96), nn.ReLU(), nn.Dropout(0.2), nn.Linear(96, 2)
        )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        counts = ids.ne(0).sum(dim=1, keepdim=True).clamp(min=1)
        averages = self.embedding(ids).sum(dim=1) / counts
        return self.classifier(averages)


class SkipGram(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.center = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.context = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)

    def forward(self, center, context, negatives):
        vectors = self.center(center)
        positives = (vectors * self.context(context)).sum(dim=1)
        negatives = (vectors[:, None, :] * self.context(negatives)).sum(dim=2)
        return -(torch.nn.functional.logsigmoid(positives) +
                 torch.nn.functional.logsigmoid(-negatives).sum(dim=1)).mean()


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


class ModelBank:
    def __init__(self, directory: Path):
        if not (directory / "manifest.json").is_file():
            raise FileNotFoundError(f"Model bundle missing at {directory}. See README.md.")
        self.tfidf = joblib.load(directory / "tfidf.joblib")
        word_package = torch.load(directory / "mlp.pt", map_location="cpu", weights_only=True)
        self.vocab = word_package["vocab"]
        self.mlp = WordMLP(len(self.vocab))
        self.mlp.load_state_dict(word_package["weights"])
        self.mlp.eval()
        self.tokenizers = {
            key: make_tokenizer(directory / key / "vocab.txt") for key in ("bert", "int8")
        }
        self.bert = self._load_bert(directory / "bert", quantized=False)
        self.int8 = self._load_bert(directory / "int8", quantized=True)

    @staticmethod
    def _load_bert(directory: Path, quantized: bool):
        config = BertConfig.from_json_file(directory / "config.json")
        config._attn_implementation = "eager"
        model = BertForSequenceClassification(config).eval()
        if quantized:
            model = quantize_bert(model)
        state = torch.load(directory / "weights.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model.eval()

    @torch.inference_mode()
    def probabilities(self, key: str, texts: list[str]) -> np.ndarray:
        if key == "tfidf":
            return self.tfidf.predict_proba(texts)
        if key == "mlp":
            logits = self.mlp(encode_words(texts, self.vocab))
        else:
            encoded = self.tokenizers[key](
                texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
            )
            logits = getattr(self, key)(**encoded).logits
        return torch.softmax(logits, dim=1).numpy()
