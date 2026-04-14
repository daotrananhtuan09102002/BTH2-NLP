import argparse
import json
import math
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from model import MAX_PARAMETER_BUDGET, NLI, collate_fn, tokenizes


ARTIFACT_VERSION = "nli_bert_wordpiece_v2"
DEFAULT_DEV_FILE = "data/processed/dev.jsonl"
LABEL_TO_ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "e": 0,
    "n": 1,
    "c": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Offline self-check for the trained NLI model.")
    parser.add_argument("--model-dir", default="MODEL")
    parser.add_argument("--data", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def normalize_label(label):
    if isinstance(label, bool):
        return None

    if isinstance(label, int):
        return label if label in (0, 1, 2) else None

    if isinstance(label, float):
        if not math.isfinite(label):
            return None
        label = int(label)
        return label if label in (0, 1, 2) else None

    if isinstance(label, str):
        label = label.strip().lower()
        if label.isdigit():
            numeric = int(label)
            return numeric if numeric in (0, 1, 2) else None
        return LABEL_TO_ID.get(label)

    return None


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}.") from error
    return rows


def normalize_example(row):
    premise = row.get("premise") or row.get("sentence1")
    hypothesis = row.get("hypothesis") or row.get("sentence2")
    label = normalize_label(row.get("label"))

    if not isinstance(premise, str) or not isinstance(hypothesis, str):
        return None

    premise = premise.strip()
    hypothesis = hypothesis.strip()
    if not premise or not hypothesis or label is None:
        return None

    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
    }


def load_examples(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    examples = []
    for row in read_jsonl(path):
        example = normalize_example(row)
        if example is not None:
            examples.append(example)

    if not examples:
        raise ValueError(f"No usable evaluation examples were found in {path}.")
    return examples


def load_training_summary(model_dir):
    summary_path = Path(model_dir) / "training_summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            "MODEL/ does not contain training_summary.json. "
            "The current artifact is likely the old LSTM starter checkpoint. "
            "Run train.py to generate a compatible BERT-based MODEL/ package."
        )

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    if summary.get("artifact_version") != ARTIFACT_VERSION:
        raise RuntimeError(
            "MODEL/ contains an unsupported artifact version. "
            "Run train.py again to rebuild MODEL/ with the new pipeline."
        )
    return summary


class NLIDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        encoded = tokenizes(
            (example["premise"], example["hypothesis"]),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        return {
            "input_ids": encoded["input_ids"],
            "labels": example["label"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "sep_token_id": self.tokenizer.sep_token_id,
        }


def compute_metrics(predictions, labels):
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            total_loss += float(outputs.loss.item())
            total_batches += 1
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

    metrics = compute_metrics(predictions, labels)
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics


def main():
    args = parse_args()
    summary = load_training_summary(args.model_dir)

    evaluation_path = args.data or summary.get("holdout_output") or DEFAULT_DEV_FILE
    examples = load_examples(evaluation_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    max_length = min(
        int(tokenizer.model_max_length),
        int(summary.get("config", {}).get("max_position_embeddings", tokenizer.model_max_length)),
    )
    dataset = NLIDataset(examples, tokenizer, max_length=max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    try:
        model = NLI.from_pretrained(args.model_dir)
    except Exception as error:
        raise RuntimeError(
            "Failed to load MODEL/ with the current architecture. "
            "This usually means MODEL/ still contains an old checkpoint or a partial training run."
        ) from error

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count >= MAX_PARAMETER_BUDGET:
        raise RuntimeError(
            f"Loaded model exceeds the 40M parameter budget: {parameter_count:,}."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    metrics = evaluate_model(model, dataloader, device)

    print(f"Evaluation file: {evaluation_path}")
    print(f"Examples: {len(examples):,}")
    print(f"Parameters: {parameter_count:,}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
