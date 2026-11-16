"""Download, deduplicate, split, train on CPU, and build deployable weights."""
import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import joblib
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification

from .models import MAX_LENGTH, VOCAB_LIMIT, SkipGram, WordMLP
from .models import encode_words, init_cpu, make_tokenizer, quantize_bert
from .text import normalize

SEED = 42
DATA_URL = ("https://raw.githubusercontent.com/t-davidson/"
            "hate-speech-and-offensive-language/master/data/labeled_data.csv")
DATA_SHA256 = "fcb8bc7c68120ae4af04a5b9acd58585513ede11e1548ebf36a5c2040b6f6281"
BERT_REPO = "google/bert_uncased_L-2_H-128_A-2"
BERT_REV = "30b0a37ccaaa32f332884b96992754e246e48c5f"


def dataset(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".download")
        urlretrieve(DATA_URL, temp)
        temp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise ValueError("Unexpected dataset checksum. Verify the pinned Davidson CSV.")
    groups = defaultdict(list)
    count = 0
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            count += 1
            text = normalize(row["tweet"])
            if text:
                groups[text].append((row[""], int(row["class"] != "2")))
    rows = [(examples[0][0], text, examples[0][1]) for text, examples in groups.items()
            if len({label for _, label in examples}) == 1]
    train, holdout = train_test_split(
        rows, test_size=0.30, random_state=SEED, stratify=[r[2] for r in rows]
    )
    val, test = train_test_split(
        holdout, test_size=0.5, random_state=SEED, stratify=[r[2] for r in holdout]
    )
    assert len({r[1] for r in train + val + test}) == len(rows)
    return (train, val, test), count, digest


def class_weights(labels):
    counts = np.bincount(labels, minlength=2)
    return torch.tensor(len(labels) / (2 * counts), dtype=torch.float32)


def train_mlp(train_text, train_y, valid_text, valid_y):
    word_counts = Counter(word for text in train_text for word in text.split())
    words = [word for word, count in word_counts.most_common(VOCAB_LIMIT - 2) if count >= 2]
    vocab = {"<pad>": 0, "<unk>": 1, **{word: i for i, word in enumerate(words, 2)}}
    # Skip-gram context window ±2, with 5 negative samples per observed pair.
    pairs = []
    for text in train_text:
        ids = [vocab.get(token, 1) for token in text.split()[:MAX_LENGTH]]
        for i, center in enumerate(ids):
            if center == 1:
                continue
            pairs.extend((center, ids[j]) for j in range(max(0, i - 2), min(len(ids), i + 3))
                         if j != i and ids[j] != 1)
    frequencies = torch.tensor([0, 0] + [word_counts[word] for word in words]).float().pow(0.75)
    sg = SkipGram(len(vocab))
    optimizer = torch.optim.SparseAdam(sg.parameters(), lr=0.025)
    # Sparse gradients update only the participating embeddings.
    sg.center.sparse = sg.context.sparse = True
    pairs_tensor = torch.tensor(pairs, dtype=torch.long)
    for epoch in range(3):
        for ids in torch.randperm(len(pairs_tensor)).split(2048):
            center, context = pairs_tensor[ids].T
            negatives = frequencies.multinomial(len(ids) * 5, replacement=True).view(-1, 5)
            optimizer.zero_grad()
            loss = sg(center, context, negatives)
            loss.backward()
            optimizer.step()
        print(f"word2vec epoch {epoch + 1} complete", flush=True)
    model = WordMLP(len(vocab))
    with torch.no_grad():
        model.embedding.weight.copy_(sg.center.weight)
        model.embedding.weight[1] = model.embedding.weight[2:].mean(dim=0)
    model.embedding.weight.requires_grad_(False)
    loader = DataLoader(
        TensorDataset(encode_words(train_text, vocab), torch.tensor(train_y)),
        batch_size=128, shuffle=True,
    )
    valid_ids = encode_words(valid_text, vocab)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_y))
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.08, momentum=0.9)
    scores = []
    state = None
    best = -1.0
    for epoch in range(18):
        model.train()
        for ids, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(ids), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            preds = model(valid_ids).argmax(dim=1)
        score = f1_score(valid_y, preds, average="macro")
        scores.append(float(score))
        if score > best:
            state = {key: value.clone() for key, value in model.state_dict().items()}
            best = score
    model.load_state_dict(state)
    return model.eval(), vocab, scores


def fine_tune(bert_source, tokenizer, train_text, train_y, valid_text, valid_y, epochs):
    model = BertForSequenceClassification.from_pretrained(
        bert_source, revision=BERT_REV, num_labels=2, attn_implementation="eager"
    ).cpu()
    def batches(texts, labels, shuffle):
        encoded = tokenizer(texts, padding="max_length", max_length=MAX_LENGTH,
                            truncation=True, return_tensors="pt")
        tensors = [encoded[key] for key in ("input_ids", "attention_mask", "token_type_ids")]
        return DataLoader(TensorDataset(*tensors, torch.tensor(labels)),
                          batch_size=48, shuffle=shuffle)
    train_batches = batches(train_text, train_y, True)
    valid_batches = batches(valid_text, valid_y, False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_y))
    best = -1.0
    best_state = None
    scores = []
    for epoch in range(epochs):
        model.train()
        for ids, mask, types, labels in train_batches:
            optimizer.zero_grad()
            logits = model(input_ids=ids, attention_mask=mask, token_type_ids=types).logits
            criterion(logits, labels).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        outputs = []
        with torch.inference_mode():
            for ids, mask, types, _ in valid_batches:
                logits = model(input_ids=ids, attention_mask=mask, token_type_ids=types).logits
                outputs.extend(logits.argmax(dim=1).tolist())
        score = f1_score(valid_y, outputs, average="macro")
        scores.append(float(score))
        if score > best:
            best = score
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
        print(f"BERT epoch {epoch + 1}: validation macro-F1 {score:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model.eval(), scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/labeled_data.csv"))
    parser.add_argument("--bert-source", default=None, help="Optional local Hugging Face snapshot")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs and threads must be positive")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    engine = init_cpu(args.threads)
    torch.manual_seed(SEED)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    (train, validation, test), raw_count, digest = dataset(args.data)
    train_text, train_y = [r[1] for r in train], [r[2] for r in train]
    valid_text, valid_y = [r[1] for r in validation], [r[2] for r in validation]
    baseline = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True),
        LogisticRegression(class_weight="balanced", max_iter=600, C=3, random_state=SEED),
    )
    baseline.fit(train_text, train_y)
    joblib.dump(baseline, output / "tfidf.joblib", compress=0)
    print(f"TF-IDF fitted, split sizes {len(train)}/{len(validation)}/{len(test)}", flush=True)
    mlp, vocab, mlp_scores = train_mlp(train_text, train_y, valid_text, valid_y)
    torch.save({"vocab": vocab, "weights": mlp.state_dict()}, output / "mlp.pt")
    bert_source = args.bert_source or BERT_REPO
    if args.bert_source:
        vocab_path = Path(args.bert_source) / "vocab.txt"
    else:
        from huggingface_hub import hf_hub_download
        vocab_path = Path(hf_hub_download(BERT_REPO, "vocab.txt", revision=BERT_REV))
    tokenizer = make_tokenizer(vocab_path)
    bert, bert_scores = fine_tune(bert_source, tokenizer, train_text, train_y,
                                 valid_text, valid_y, args.epochs)
    for name, state in (("bert", bert.state_dict()), ("int8", quantize_bert(bert).state_dict())):
        folder = output / name
        folder.mkdir(exist_ok=True)
        bert.config.to_json_file(folder / "config.json")
        shutil.copyfile(vocab_path, folder / "vocab.txt")
        torch.save(state, folder / "weights.pt")
    split_info = {
        label: {"total": len(rows), "clean": sum(r[2] == 0 for r in rows),
                "offensive": sum(r[2] == 1 for r in rows)}
        for label, rows in (("train", train), ("validation", validation), ("test", test))
    }
    manifest = {
        "raw_rows": raw_count, "dataset_sha256": digest, "seed": SEED, "splits": split_info,
        "bert_repo": BERT_REPO, "bert_revision": BERT_REV, "threads": args.threads,
        "quantization_engine": engine, "mlp_validation_macro_f1": mlp_scores,
        "bert_validation_macro_f1": bert_scores,
        "test_ids_sha256": hashlib.sha256(",".join(r[0] for r in test).encode()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Training and quantization complete", flush=True)


if __name__ == "__main__":
    main()
