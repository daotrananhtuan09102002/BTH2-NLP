import argparse
import copy
import json
import math
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from tokenizers import Tokenizer, normalizers, processors
from tokenizers.models import WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.trainers import WordPieceTrainer
from torch.utils.data import DataLoader, Dataset
from transformers import BertForMaskedLM, PreTrainedTokenizerFast, get_linear_schedule_with_warmup

from model import (
    DEFAULT_CLS_TOKEN_ID,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MASK_TOKEN_ID,
    DEFAULT_PAD_TOKEN_ID,
    DEFAULT_SEP_TOKEN_ID,
    DEFAULT_UNK_TOKEN_ID,
    MAX_PARAMETER_BUDGET,
    NLI,
    NLIConfig,
    collate_fn,
    estimated_parameter_count,
    tokenizes,
)


ARTIFACT_VERSION = "nli_bert_wordpiece_v2"
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
    parser = argparse.ArgumentParser(description="Train a near-40M English NLI model offline.")
    parser.add_argument("--train-files", nargs="+", default=DEFAULT_TRAIN_FILES)
    parser.add_argument("--model-dir", default="MODEL")
    parser.add_argument("--holdout-output", default=DEFAULT_HOLDOUT_FILE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--holdout-ratio", type=float, default=0.015)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--disable-gradient-checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.add_argument("--save-every-epoch", action="store_true")

    parser.add_argument("--vocab-size", type=int, default=26_000)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--classifier-hidden-size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--mlm-epochs", type=int, default=1)
    parser.add_argument("--mlm-batch-size", type=int, default=8)
    parser.add_argument("--mlm-grad-accum", type=int, default=16)
    parser.add_argument("--mlm-lr", type=float, default=5e-4)
    parser.add_argument("--mlm-probability", type=float, default=0.15)

    parser.add_argument("--nli-epochs", type=int, default=4)
    parser.add_argument("--nli-batch-size", type=int, default=8)
    parser.add_argument("--nli-grad-accum", type=int, default=8)
    parser.add_argument("--nli-lr", type=float, default=2e-4)
    parser.add_argument("--anli-oversample", type=int, default=3)
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
            f"Checked: {', '.join(paths)}. "
            "Prepare local data first with prepare_data.py or pass --train-files explicitly."
        )

    return all_examples, used_files


def build_holdout_split(examples, holdout_ratio, seed):
    labels = [example["label"] for example in examples]
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=holdout_ratio, random_state=seed)
    train_indices, holdout_indices = next(splitter.split(examples, labels))
    train_examples = [examples[index] for index in train_indices]
    holdout_examples = [examples[index] for index in holdout_indices]
    return train_examples, holdout_examples


def yield_training_text(examples):
    for example in examples:
        yield example["premise"]
        yield example["hypothesis"]


def build_tokenizer(examples, args):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", DEFAULT_CLS_TOKEN_ID),
            ("[SEP]", DEFAULT_SEP_TOKEN_ID),
        ],
    )

    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        min_frequency=2,
    )
    tokenizer.train_from_iterator(yield_training_text(examples), trainer=trainer)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=args.max_length,
        clean_up_tokenization_spaces=False,
    )
    hf_tokenizer.padding_side = "right"
    hf_tokenizer.truncation_side = "right"
    return hf_tokenizer


def build_model_config(tokenizer, args, label_smoothing):
    config = NLIConfig(
        vocab_size=len(tokenizer),
        hidden_size=args.hidden_size,
        nclass=3,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        max_position_embeddings=args.max_length,
        type_vocab_size=2,
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        mask_token_id=tokenizer.mask_token_id,
        classifier_hidden_size=args.classifier_hidden_size,
        classifier_dropout=args.dropout,
        label_smoothing=label_smoothing,
    )
    estimated_params = estimated_parameter_count(config)
    if estimated_params >= MAX_PARAMETER_BUDGET:
        raise ValueError(
            f"Configured model exceeds the 40M parameter budget: {estimated_params:,}."
        )
    return config


class PairTextDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        return {
            "premise": example["premise"],
            "hypothesis": example["hypothesis"],
        }


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


class MLMCollator:
    def __init__(self, tokenizer, max_length, mlm_probability):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        premises = [item["premise"] for item in batch]
        hypotheses = [item["hypothesis"] for item in batch]
        encoded = self.tokenizer(
            premises,
            text_pair=hypotheses,
            truncation="longest_first",
            max_length=self.max_length,
            padding="max_length",
            return_special_tokens_mask=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"]
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability, dtype=torch.float)
        special_tokens_mask = encoded.pop("special_tokens_mask").bool()
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        probability_matrix.masked_fill_(input_ids.eq(self.tokenizer.pad_token_id), 0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, dtype=torch.float)).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, dtype=torch.float)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        encoded["input_ids"] = input_ids
        encoded["labels"] = labels
        return encoded


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name] = parameter.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.backup[name] = parameter.detach().clone()
            parameter.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            parameter.data.copy_(self.backup[name])
        self.backup = {}


def create_amp_components(device, prefer_bf16=True):
    if device.type != "cuda":
        return None, None

    use_bf16 = prefer_bf16 and torch.cuda.is_bf16_supported()
    if use_bf16:
        return torch.bfloat16, None

    scaler = torch.cuda.amp.GradScaler()
    return torch.float16, scaler


def move_batch_to_device(batch, device):
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def get_autocast_context(device, amp_dtype):
    if device.type != "cuda" or amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


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
            total_loss += float(outputs.loss.item())
            total_batches += 1
            predictions.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())

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


def maybe_enable_gradient_checkpointing(model, enabled):
    if not enabled:
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "bert") and hasattr(model.bert, "gradient_checkpointing_enable"):
        model.bert.gradient_checkpointing_enable()


def run_mlm_pretraining(model, dataloader, args, device):
    maybe_enable_gradient_checkpointing(model, args.gradient_checkpointing)
    optimizer = build_optimizer(model, lr=args.mlm_lr, weight_decay=args.weight_decay)
    total_steps = count_update_steps(len(dataloader), args.mlm_grad_accum, args.mlm_epochs)
    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    amp_dtype, scaler = create_amp_components(device)

    model.to(device)
    model.train()
    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.mlm_epochs):
        running_loss = 0.0
        for batch_index, batch in enumerate(dataloader, start=1):
            batch = move_batch_to_device(batch, device)
            with get_autocast_context(device, amp_dtype):
                outputs = model(**batch)
                loss = outputs.loss / args.mlm_grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(loss.item()) * args.mlm_grad_accum

            should_step = (
                batch_index % args.mlm_grad_accum == 0
                or batch_index == len(dataloader)
            )
            if not should_step:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        average_loss = running_loss / max(len(dataloader), 1)
        print(f"[MLM] epoch={epoch + 1}/{args.mlm_epochs} loss={average_loss:.4f} steps={global_step}")

    return model


def evaluate_with_ema(model, dataloader, device, ema):
    if ema is None:
        return evaluate_nli(model, dataloader, device)

    ema.apply_shadow(model)
    try:
        return evaluate_nli(model, dataloader, device)
    finally:
        ema.restore(model)


def run_supervised_training(
    model,
    train_loader,
    eval_loader,
    args,
    device,
    max_train_steps=None,
    save_prefix=None,
):
    maybe_enable_gradient_checkpointing(model, args.gradient_checkpointing)
    optimizer = build_optimizer(model, lr=args.nli_lr, weight_decay=args.weight_decay)

    if max_train_steps is None:
        total_steps = count_update_steps(len(train_loader), args.nli_grad_accum, args.nli_epochs)
    else:
        total_steps = max_train_steps

    warmup_steps = max(int(total_steps * args.warmup_ratio), 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    amp_dtype, scaler = create_amp_components(device)
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)

    model.to(device)
    global_step = 0
    best_step = 0
    best_metrics = None

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.nli_epochs):
        model.train()
        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            with get_autocast_context(device, amp_dtype):
                outputs = model(**batch)
                loss = outputs.loss / args.nli_grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += float(loss.item()) * args.nli_grad_accum

            should_step = (
                batch_index % args.nli_grad_accum == 0
                or batch_index == len(train_loader)
            )
            if not should_step:
                continue

            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            global_step += 1

            if max_train_steps is not None and global_step >= max_train_steps:
                break

        average_train_loss = running_loss / max(len(train_loader), 1)
        log_message = f"[NLI] epoch={epoch + 1}/{args.nli_epochs} train_loss={average_train_loss:.4f} steps={global_step}"

        if eval_loader is not None:
            metrics = evaluate_with_ema(model, eval_loader, device, ema)
            log_message += (
                f" eval_loss={metrics['loss']:.4f}"
                f" macro_f1={metrics['macro_f1']:.4f}"
                f" accuracy={metrics['accuracy']:.4f}"
            )
            if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
                best_metrics = metrics
                best_step = global_step

        print(log_message)

        if args.save_every_epoch and save_prefix:
            snapshot_dir = Path(args.model_dir) / f"{save_prefix}_epoch_{epoch + 1}"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            ema.apply_shadow(model)
            try:
                model.save_pretrained(snapshot_dir)
            finally:
                ema.restore(model)

        if max_train_steps is not None and global_step >= max_train_steps:
            break

    if best_step == 0:
        best_step = global_step

    return {
        "best_step": best_step,
        "best_metrics": best_metrics,
        "ema": ema,
    }


def oversample_examples(examples, factor):
    if factor <= 1:
        return list(examples)

    expanded = []
    for example in examples:
        repeats = factor if "anli" in example["source"] else 1
        expanded.extend(copy.deepcopy(example) for _ in range(repeats))
    return expanded


def save_training_summary(path, summary):
    ensure_parent_dir(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def build_data_loader(dataset, batch_size, shuffle, collate, args, generator=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        generator=generator,
    )


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

    tokenizer = build_tokenizer(full_examples, args)
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(args.model_dir)

    model_config = build_model_config(tokenizer, args, label_smoothing=args.label_smoothing)
    estimated_params = estimated_parameter_count(model_config)
    print(f"Tokenizer vocab size: {len(tokenizer):,}")
    print(f"Estimated model parameters: {estimated_params:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    mlm_dataset = PairTextDataset(full_examples)
    mlm_loader = build_data_loader(
        mlm_dataset,
        batch_size=args.mlm_batch_size,
        shuffle=True,
        collate=MLMCollator(
            tokenizer=tokenizer,
            max_length=args.max_length,
            mlm_probability=args.mlm_probability,
        ),
        args=args,
        generator=torch.Generator().manual_seed(args.seed),
    )

    mlm_model = BertForMaskedLM(model_config.to_bert_config())
    mlm_model = run_mlm_pretraining(mlm_model, mlm_loader, args, device)
    pretrained_encoder_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in mlm_model.bert.state_dict().items()
    }
    del mlm_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    supervised_train_examples = oversample_examples(train_examples, args.anli_oversample)
    supervised_train_dataset = NLIDataset(supervised_train_examples, tokenizer, args.max_length)
    holdout_dataset = NLIDataset(holdout_examples, tokenizer, args.max_length)

    train_loader = build_data_loader(
        supervised_train_dataset,
        batch_size=args.nli_batch_size,
        shuffle=True,
        collate=collate_fn,
        args=args,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    holdout_loader = build_data_loader(
        holdout_dataset,
        batch_size=args.nli_batch_size,
        shuffle=False,
        collate=collate_fn,
        args=args,
    )

    holdout_model = NLI(model_config)
    holdout_model.bert.load_state_dict(pretrained_encoder_state, strict=True)
    holdout_result = run_supervised_training(
        model=holdout_model,
        train_loader=train_loader,
        eval_loader=holdout_loader,
        args=args,
        device=device,
        save_prefix="holdout",
    )

    best_step = holdout_result["best_step"]
    best_metrics = holdout_result["best_metrics"] or {}
    print(f"Best holdout step budget: {best_step}")
    del holdout_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    full_supervised_examples = oversample_examples(full_examples, args.anli_oversample)
    final_train_dataset = NLIDataset(full_supervised_examples, tokenizer, args.max_length)
    final_train_loader = build_data_loader(
        final_train_dataset,
        batch_size=args.nli_batch_size,
        shuffle=True,
        collate=collate_fn,
        args=args,
        generator=torch.Generator().manual_seed(args.seed + 2),
    )

    final_model = NLI(model_config)
    final_model.bert.load_state_dict(pretrained_encoder_state, strict=True)
    final_result = run_supervised_training(
        model=final_model,
        train_loader=final_train_loader,
        eval_loader=None,
        args=args,
        device=device,
        max_train_steps=best_step,
        save_prefix=None,
    )

    final_result["ema"].apply_shadow(final_model)
    try:
        final_model.save_pretrained(args.model_dir)
    finally:
        final_result["ema"].restore(final_model)

    summary = {
        "artifact_version": ARTIFACT_VERSION,
        "train_files": used_files,
        "holdout_output": args.holdout_output,
        "num_full_examples": len(full_examples),
        "num_train_examples": len(train_examples),
        "num_holdout_examples": len(holdout_examples),
        "source_distribution": Counter(example["source"] for example in full_examples),
        "label_distribution": Counter(str(example["label"]) for example in full_examples),
        "tokenizer_vocab_size": len(tokenizer),
        "estimated_parameters": estimated_params,
        "best_step": best_step,
        "best_holdout_metrics": best_metrics,
        "config": model_config.to_dict(),
    }
    save_training_summary(Path(args.model_dir) / "training_summary.json", summary)

    print("Training completed.")
    print(json.dumps(summary, indent=2, default=dict))


if __name__ == "__main__":
    main()
