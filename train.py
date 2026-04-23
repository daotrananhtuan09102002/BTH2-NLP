import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


ARTIFACT_VERSION = "nli_albert_large_v2_ft_v1"
LABEL_TO_ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "e": 0,
    "n": 1,
    "c": 2,
}
DEFAULT_TRAIN_FILES = [
    "data/processed/mnli_train.jsonl",
    "data/processed/mnli_validation_mismatched.jsonl",
    "data/processed/snli_train.jsonl",
    "data/processed/snli_validation.jsonl",
    "data/processed/snli_test.jsonl",
    "data/processed/anli_train.jsonl",
]
DEFAULT_HOLDOUT_FILE = "data/processed/dev.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune albert-large-v2 for English NLI."
    )
    parser.add_argument("--train-files", nargs="+", default=DEFAULT_TRAIN_FILES)
    parser.add_argument("--model-name", default="albert-large-v2")
    parser.add_argument("--model-dir", default="MODEL")
    parser.add_argument("--holdout-output", default=DEFAULT_HOLDOUT_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--holdout-ratio", type=float, default=0.015)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--multi-gpu", action="store_true")

    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--nli-epochs", type=int, default=2)
    parser.add_argument("--nli-batch-size", type=int, default=16)
    parser.add_argument("--nli-grad-accum", type=int, default=1)
    parser.add_argument("--nli-lr", type=float, default=2e-5)
    parser.add_argument("--anli-oversample", type=int, default=1)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}.") from error
    return rows


def write_jsonl(path, rows):
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def infer_source(path, row):
    if isinstance(row.get("source"), str) and row["source"].strip():
        return row["source"].strip().lower()
    return Path(path).stem.lower()


def normalize_example(row, source):
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
        "source": source,
    }


def load_examples(paths):
    all_examples = []
    seen = set()
    used_files = []

    for path in paths:
        rows = read_jsonl(path)
        if not rows:
            continue

        used_files.append(path)
        for row in rows:
            source = infer_source(path, row)
            example = normalize_example(row, source)
            if example is None:
                continue

            dedupe_key = (example["premise"], example["hypothesis"], example["label"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            all_examples.append(example)

    if not all_examples:
        raise FileNotFoundError(
            "No usable JSONL training examples were found. "
            f"Checked: {', '.join(paths)}."
        )

    return all_examples, used_files


def build_holdout_split(examples, holdout_ratio, seed):
    labels = [example["label"] for example in examples]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=seed)
    train_indices, holdout_indices = next(splitter.split(examples, labels))
    train_examples = [examples[index] for index in train_indices]
    holdout_examples = [examples[index] for index in holdout_indices]
    return train_examples, holdout_examples


def oversample_examples(examples, factor):
    if factor <= 1:
        return list(examples)

    expanded = []
    for example in examples:
        repeats = factor if "anli" in example["source"] else 1
        expanded.extend(example.copy() for _ in range(repeats))
    return expanded


class NLIDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        encoded = self.tokenizer(
            example["premise"],
            text_pair=example["hypothesis"],
            truncation="longest_first",
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "token_type_ids": encoded.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0)
            if encoded.get("token_type_ids") is not None
            else torch.zeros(self.max_length, dtype=torch.long),
            "labels": torch.tensor(example["label"], dtype=torch.long),
        }


def collate_nli(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch], dim=0),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch], dim=0),
        "token_type_ids": torch.stack([item["token_type_ids"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
    }


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
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


def unwrap_model(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module
    return model


def evaluate_nli(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            total_loss += float(loss.item())
            total_batches += 1
            logits = outputs.logits
            predictions.extend(logits.argmax(dim=-1).detach().cpu().tolist())
            labels.extend(batch["labels"].detach().cpu().tolist())

    metrics = compute_metrics(predictions, labels)
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics


def build_optimizer(model, lr, weight_decay):
    no_decay = {"bias", "LayerNorm.weight"}
    named_parameters = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": [
                parameter
                for name, parameter in named_parameters
                if parameter.requires_grad and not any(term in name for term in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in named_parameters
                if parameter.requires_grad and any(term in name for term in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)


def count_update_steps(num_batches, grad_accum, epochs):
    steps_per_epoch = math.ceil(num_batches / max(grad_accum, 1))
    return max(steps_per_epoch * epochs, 1)


def run_supervised_training(model, train_loader, eval_loader, args, device):
    model.to(device)
    optimizer = build_optimizer(model, lr=args.nli_lr, weight_decay=args.weight_decay)

    total_steps = count_update_steps(len(train_loader), args.nli_grad_accum, args.nli_epochs)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    global_step = 0
    best_step = 0
    best_metrics = None
    log_interval = max(int(args.log_interval), 1)

    print(
        "[NLI] start "
        f"epochs={args.nli_epochs} batches_per_epoch={len(train_loader)} "
        f"grad_accum={args.nli_grad_accum} total_update_steps={total_steps}",
        flush=True,
    )

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.nli_epochs):
        model.train()
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            loss = outputs.loss
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            loss = loss / args.nli_grad_accum

            loss.backward()
            running_loss += float(loss.item()) * args.nli_grad_accum

            should_step = (
                batch_index % args.nli_grad_accum == 0
                or batch_index == len(train_loader)
            )
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % log_interval == 0 or global_step == total_steps:
                avg_seen_loss = running_loss / max(batch_index, 1)
                print(
                    f"[NLI] step={global_step}/{total_steps} "
                    f"epoch={epoch + 1}/{args.nli_epochs} "
                    f"batch={batch_index}/{len(train_loader)} "
                    f"avg_loss={avg_seen_loss:.4f}",
                    flush=True,
                )

        average_train_loss = running_loss / max(len(train_loader), 1)
        metrics = evaluate_nli(model, eval_loader, device)
        print(
            f"[NLI] epoch={epoch + 1}/{args.nli_epochs} "
            f"train_loss={average_train_loss:.4f} "
            f"eval_loss={metrics['loss']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"accuracy={metrics['accuracy']:.4f} "
            f"steps={global_step}",
            flush=True,
        )

        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_step = global_step
            unwrap_model(model).save_pretrained(args.model_dir)

        if args.save_every_epoch:
            snapshot_dir = Path(args.model_dir) / f"epoch_{epoch + 1}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            unwrap_model(model).save_pretrained(snapshot_dir)

    if best_step == 0:
        best_step = global_step

    return {
        "best_step": best_step,
        "best_metrics": best_metrics,
    }


def build_data_loader(dataset, batch_size, shuffle, args, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_nli,
        generator=generator,
    )


def save_training_summary(path, summary):
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)

    full_examples, used_files = load_examples(args.train_files)
    train_examples, holdout_examples = build_holdout_split(
        full_examples,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    write_jsonl(args.holdout_output, holdout_examples)

    supervised_train_examples = oversample_examples(train_examples, args.anli_oversample)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.model_max_length = args.max_length
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.model_dir)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        label2id={"entailment": 0, "neutral": 1, "contradiction": 2},
    )

    estimated_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Pretrained model: {args.model_name}")
    print(f"Model parameters: {estimated_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_data_parallel = args.multi_gpu and gpu_count > 1
    print(f"Visible CUDA devices: {gpu_count}")
    if args.multi_gpu and gpu_count <= 1:
        print("[MultiGPU] --multi-gpu was set but only one CUDA device is available.", flush=True)
    if use_data_parallel:
        print(f"[MultiGPU] DataParallel enabled across {gpu_count} GPUs.", flush=True)
        model = torch.nn.DataParallel(model)

    train_dataset = NLIDataset(supervised_train_examples, tokenizer, args.max_length)
    holdout_dataset = NLIDataset(holdout_examples, tokenizer, args.max_length)

    train_loader = build_data_loader(
        train_dataset,
        batch_size=args.nli_batch_size,
        shuffle=True,
        args=args,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    holdout_loader = build_data_loader(
        holdout_dataset,
        batch_size=args.nli_batch_size,
        shuffle=False,
        args=args,
    )
    print(
        f"[Data] train examples={len(train_dataset):,} batches={len(train_loader):,} "
        f"holdout examples={len(holdout_dataset):,} batches={len(holdout_loader):,}",
        flush=True,
    )

    result = run_supervised_training(
        model=model,
        train_loader=train_loader,
        eval_loader=holdout_loader,
        args=args,
        device=device,
    )

    summary = {
        "artifact_version": ARTIFACT_VERSION,
        "model_name": args.model_name,
        "train_files": used_files,
        "holdout_output": args.holdout_output,
        "num_full_examples": len(full_examples),
        "num_train_examples": len(supervised_train_examples),
        "num_holdout_examples": len(holdout_examples),
        "source_distribution": Counter(example["source"] for example in full_examples),
        "label_distribution": Counter(str(example["label"]) for example in full_examples),
        "max_length": args.max_length,
        "model_parameters": estimated_params,
        "best_step": result["best_step"],
        "best_holdout_metrics": result["best_metrics"] or {},
    }
    save_training_summary(Path(args.model_dir) / "training_summary.json", summary)

    print("Training completed.")
    print(json.dumps(summary, indent=2, default=dict))


if __name__ == "__main__":
    main()
