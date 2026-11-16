"""Evaluate all four models on one unchanged test set and benchmark CPU inference."""
import argparse
import csv
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import sklearn
import torch
import transformers
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from .app import Message, moderate_text
from .models import MODEL_NAMES, ModelBank, init_cpu, size_bytes
from .text import normalize, rule_label
from .train import dataset

DEMO_INPUTS = (
    "Good game, everyone. Let's try the left route next round.",
    "You are a fukking idiot, uninstall the game.",
    "I'm quoting the player who called me stupid so I can report the message.",
)
CPU_BUDGET_MS = 50


def describe_hardware():
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        info = subprocess.run(
            ["system_profiler", "SPHardwareDataType"], capture_output=True, text=True, check=False
        ).stdout
        match = re.search(r"^\s*Chip:\s*(.+)$", info, flags=re.M)
        if match:
            cpu = match[1]
    elif Path("/proc/cpuinfo").exists():
        match = re.search(r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.M)
        if match:
            cpu = match[1]
    return {"cpu": cpu, "architecture": platform.machine(), "os": platform.platform()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/labeled_data.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    if args.runs < 100 or args.threads < 1:
        parser.error("Use at least 100 timed inputs and a positive thread count.")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    engine = init_cpu(args.threads)
    models = ModelBank(args.artifacts)
    (_, _, test), _, digest = dataset(args.data)
    sample = np.random.default_rng(42).choice(len(test), min(args.runs, len(test)), replace=False)
    manifest = json.loads((args.artifacts / "manifest.json").read_text())
    if digest != manifest["dataset_sha256"]:
        raise ValueError("Dataset differs from the training manifest.")
    test_ids = ",".join(row[0] for row in test).encode()
    if hashlib.sha256(test_ids).hexdigest() != manifest["test_ids_sha256"]:
        raise ValueError("Held-out partition differs from the training manifest.")
    texts = [row[1] for row in test]
    labels = [row[2] for row in test]
    rows = []
    predictions = {}
    examples = {}
    errors = {}
    for key, name in MODEL_NAMES.items():
        probs = np.vstack([models.probabilities(key, texts[i:i + 128])
                           for i in range(0, len(texts), 128)])
        preds = probs.argmax(axis=1)
        predictions[key] = preds
        for text in texts[:25]:
            models.probabilities(key, [text])
        timings = []
        for index in sample:
            start = perf_counter_ns()
            models.probabilities(key, [texts[index]])
            timings.append((perf_counter_ns() - start) / 1_000_000)
        target = args.artifacts / {"tfidf": "tfidf.joblib", "mlp": "mlp.pt"}.get(key, key)
        row = {
            "key": key, "model": name, "macro_f1": float(f1_score(labels, preds, average="macro")),
            "accuracy": float(accuracy_score(labels, preds)),
            "size_bytes": size_bytes(target), "mean_ms": statistics.fmean(timings),
            "p95_ms": float(np.percentile(timings, 95)), "runs": len(timings),
            "confusion": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
        examples[key] = [
            {"message": raw, "label": "offensive" if scores.argmax() else "clean",
             "confidence": float(scores.max())}
            for raw, scores in zip(DEMO_INPUTS,
                                   models.probabilities(key, [normalize(x) for x in DEMO_INPUTS]))
        ]
        if key == "int8":
            for expected, predicted, name in ((0, 1, "false_positive"), (1, 0, "false_negative")):
                candidates = [i for i, label in enumerate(labels) if label == expected
                              and preds[i] == predicted and "usertoken" not in texts[i]
                              and "urltoken" not in texts[i] and 6 <= len(texts[i].split()) <= 14]
                if candidates:
                    i = max(candidates, key=lambda index: probs[index, predicted])
                    errors[name] = {"id": test[i][0],
                                    "confidence": float(probs[i, predicted])}
    # Measure the live processing path on the original held-out text, within the API limit.
    heldout_ids = {row[0] for row in test}
    with args.data.open(encoding="utf-8", newline="") as source:
        raw_by_id = {row[""]: row["tweet"] for row in csv.DictReader(source)
                     if row[""] in heldout_ids and 1 <= len(row["tweet"]) <= 600}
    live_ids = [row[0] for row in test if row[0] in raw_by_id]
    live_sample = np.random.default_rng(42).choice(
        live_ids, min(args.runs, len(live_ids)), replace=False
    )
    live_times = [moderate_text(models, Message(text=raw_by_id[record_id]))["latency_ms"]
                  for record_id in live_sample]
    lexicon_predictions = [rule_label(text) for text in texts]
    payload = {
        "models": rows, "manifest": manifest, "hardware": describe_hardware(),
        "benchmark": {"threads": args.threads, "engine": engine, "batch_size": 1,
                      "warmups_per_model": 25, "latency_budget_ms": CPU_BUDGET_MS,
                      "runtime_versions": {"python": platform.python_version(),
                                           "torch": torch.__version__,
                                           "transformers": transformers.__version__,
                                           "scikit_learn": sklearn.__version__}},
        "live_int8": {"runs": len(live_times), "mean_ms": statistics.fmean(live_times),
                      "p95_ms": float(np.percentile(live_times, 95)),
                      "max_ms": max(live_times),
                      "over_budget": sum(ms >= CPU_BUDGET_MS for ms in live_times)},
        "lexicon_macro_f1": float(f1_score(labels, lexicon_predictions, average="macro")),
        "fp32_int8_agreement": float(np.mean(predictions["bert"] == predictions["int8"])),
        "examples": examples, "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
