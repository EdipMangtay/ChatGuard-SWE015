# ChatGuard

SWE015 Homework · Use Case 4 · Real-time moderation for English gaming chat.

ChatGuard classifies newly typed messages, masks messages it flags, and displays the selected model, confidence and processing time. BERT Tiny INT8 is the default. The interface also lets you try the other three models and displays an independent edit-distance lexicon flag.

## Setup and run

Install Python 3.12 and run these commands in the repository root. In Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Build the four model checkpoints from the pinned public training data and BERT initialization:

```sh
python -m chatguard.train --threads 4
```

If you have the accompanying `ChatGuard_SWE015_Model_Agirliklari.zip`, extract it in the repository root to use the measured checkpoints immediately. Confirm that `artifacts/manifest.json` exists before starting the server.

Run the application with one command:

```sh
python -m uvicorn chatguard.app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Try “Good game, everyone. Let's try the left route next round.” and “You are a fukking idiot, uninstall the game.” Try the second example again with the TF-IDF model selected. The app keeps the chat log in the browser only.

## Reproduce the experiment

```sh
python -m chatguard.train --threads 4
python -m chatguard.evaluate --threads 2 --runs 200
python -m unittest discover -s tests -v
```

The trainer downloads the pinned public CSV and BERT checkpoint on first use. For local copies, pass `--data path/to/labeled_data.csv --bert-source path/to/bert_snapshot`. The snapshot needs `config.json`, `vocab.txt` and `model.safetensors`. Training uses CPU, draws only on the train partition and validates the MLP and BERT on the validation partition. The evaluator writes the measured values, confusion matrices, validation histories, dataset checksum and held-out ID fingerprint to `results/metrics.json`. Runtime and library changes can affect reproducibility and timings.

The CSV contains 24,783 rows. Normalization, duplicate removal and a stratified fixed-seed split yield 17,056 train, 3,655 validation and 3,655 test rows. Label 2 is clean and labels 0 and 1 are offensive. The rule baseline uses a small handwritten lexicon with Levenshtein distance ≤ 1. The MLP uses self-trained skip-gram vectors. The task uses English text, so a compact English BERT checkpoint is the encoder. Training fine-tunes all encoder weights and the task head. The INT8 build stores per-row symmetric signed codes for every linear and embedding weight. LayerNorm, bias and activation computation remain FP32. This weight-only method does not assume an INT8 arithmetic kernel.

## Measured results

| Model | Macro-F1 | Serialized MB | Mean ms | P95 ms |
| --- | ---: | ---: | ---: | ---: |
| TF-IDF + logistic regression | 0.9158 | 1.33 | 0.193 | 0.210 |
| Word2vec + feedforward | 0.7364 | 2.17 | 0.024 | 0.026 |
| BERT Tiny FP32 | 0.9177 | 17.79 | 0.466 | 0.532 |
| BERT Tiny INT8 | 0.9176 | 4.78 | 0.649 | 0.738 |

On an Apple M3 Pro, each latency row uses 200 identical held-out single-message inputs, two CPU threads, batch size one and 25 warm-ups per model. Table timings include model-specific vectorization or tokenization and inference. A separate profile of the live INT8 processing path includes normalization, lexicon lookup and masking. The live path averaged 0.734 ms, with a P95 of 0.920 ms and a maximum of 1.263 ms across 200 messages. HTTP transport and browser rendering are outside both timing scopes. Model sizes use decimal MB and include inference files for each model. INT8 reduced serialized BERT size by 73.1% while adding 0.182 ms mean model latency. The live processing budget was 50 ms per message.

The INT8 model labeled held-out row #14033 clean despite a gold offensive label. The text attacks several groups using slurs and the model assigned 98.1% confidence to its wrong prediction. The benchmark and exact class counts are available in [results/metrics.json](results/metrics.json). A second run on the same held-out set [confirmed the same model sizes and scores](results/verification_run.json). The presentation reports the original run's timings.

## Code map

| File | Purpose |
| --- | --- |
| `chatguard/text.py` | Unicode and regex normalization, edit distance and lexicon |
| `chatguard/train.py` | Dataset, split, TF-IDF, word2vec, MLP and fine-tuning |
| `chatguard/quantize.py` | Weight-only INT8 conversion and inference modules |
| `chatguard/models.py` | Model loading and prediction |
| `chatguard/evaluate.py` | Paired held-out quality and timing measurement |
| `chatguard/app.py` and `chatguard/web/` | FastAPI routes and chat interface |
| `tests/test_app.py` | Normalization, masking, model-switching, artifact and route checks |

## Sources and limits

- Davidson et al. (2017), [dataset repository](https://github.com/t-davidson/hate-speech-and-offensive-language) and [paper](https://doi.org/10.1609/icwsm.v11i1.14955).
- Google [BERT Tiny checkpoint](https://huggingface.co/google/bert_uncased_L-2_H-128_A-2), pinned to revision `30b0a37ccaaa32f332884b96992754e246e48c5f`. Devlin et al., [BERT](https://arxiv.org/abs/1810.04805).
- Mikolov et al., [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546).

The training data contains offensive language and comes from a social platform, which makes its distribution different from in-game chat. Some quoted insults may be flagged and some offensive posts are missed. Confidence is an uncalibrated model score. The experiment does not measure demographic fairness.

AI disclosure: OpenAI ChatGPT/Codex assisted with coding, debugging, the architecture diagram, slides and a draft narration script. The benchmark comes from the local training and evaluation runs recorded in this repository.
