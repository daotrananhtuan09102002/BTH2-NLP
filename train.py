import evaluate
import numpy as np
import torch
import argparse
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from model import NLI, collate_fn


MODEL_NAME = "albert-large-v2"
MODEL_DIR = "./MODEL"
MAX_LENGTH = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ALBERT large v2 for NLI")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--train-samples", type=int, default=0)
    parser.add_argument("--val-samples", type=int, default=0)
    parser.add_argument("--map-batch-size", type=int, default=1000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--train-head-only", action="store_true")
    return parser.parse_args()


def compute_metrics(results):
    pred, targ = results
    pred = np.argmax(pred, axis=-1)
    res = {}

    metric = evaluate.load("accuracy")
    res["accuracy"] = metric.compute(
        predictions=pred, references=targ)["accuracy"]

    metric = evaluate.load("precision")
    res["precision"] = metric.compute(
        predictions=pred,
        references=targ,
        average="macro",
        zero_division=0,
    )["precision"]

    metric = evaluate.load("recall")
    res["recall"] = metric.compute(
        predictions=pred,
        references=targ,
        average="macro",
        zero_division=0,
    )["recall"]

    metric = evaluate.load("f1")
    res["f1"] = metric.compute(
        predictions=pred,
        references=targ,
        average="macro",
    )["f1"]

    return res


def main():
    args_in = parse_args()
    model_name = args_in.model_name
    model_dir = args_in.model_dir
    max_length = args_in.max_length
    epochs = args_in.epochs
    train_batch_size = args_in.train_batch_size
    eval_batch_size = args_in.eval_batch_size
    grad_accum = args_in.grad_accum
    learning_rate = args_in.learning_rate
    weight_decay = args_in.weight_decay
    logging_steps = args_in.logging_steps
    train_samples = args_in.train_samples
    val_samples = args_in.val_samples
    map_batch_size = args_in.map_batch_size

    if args_in.quick:
        # Quick mode for sanity checks and faster iteration.
        max_length = 128
        epochs = 1.0
        train_batch_size = 8
        eval_batch_size = 16
        grad_accum = 1
        train_samples = 50000 if train_samples <= 0 else train_samples
        val_samples = 3000 if val_samples <= 0 else val_samples

    print("Loading MNLI datasets...")
    mnli = load_dataset("nyu-mll/multi_nli")

    train_ds = mnli["train"].filter(lambda x: x["label"] in (0, 1, 2))
    val_ds = mnli["validation_matched"].filter(
        lambda x: x["label"] in (0, 1, 2))

    if train_samples > 0:
        train_samples = min(train_samples, len(train_ds))
        train_ds = train_ds.select(range(train_samples))
    if val_samples > 0:
        val_samples = min(val_samples, len(val_ds))
        val_ds = val_ds.select(range(val_samples))

    print("Loading ALBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tqdm.write("Tokenizing datasets (tqdm progress from datasets.map)...")

    def preprocess(batch):
        encoded = tokenizer(
            batch["premise"],
            text_pair=batch["hypothesis"],
            truncation="longest_first",
            max_length=max_length,
            padding="max_length",
        )
        return {
            "input_ids": encoded["input_ids"],
            "labels": batch["label"],
            "pad_token_id": [tokenizer.pad_token_id] * len(batch["label"]),
            "sep_token_id": [tokenizer.sep_token_id] * len(batch["label"]),
        }

    train_set = train_ds.map(
        preprocess,
        batched=True,
        batch_size=map_batch_size,
        desc="Tokenizing train",
        remove_columns=train_ds.column_names,
    )
    val_set = val_ds.map(
        preprocess,
        batched=True,
        batch_size=map_batch_size,
        desc="Tokenizing validation",
        remove_columns=val_ds.column_names,
    )

    model = NLI.from_albert_pretrained(
        model_name, num_labels=3, classifier_dropout=0.1)
    model.config.use_cache = False

    if args_in.train_head_only:
        for parameter in model.albert.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        print("Mode: train head only (ALBERT backbone frozen)")
    else:
        print("Mode: full fine-tuning (ALBERT + classifier)")

    allparams = sum(p.numel() for p in model.parameters())
    trainparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("All Param:", allparams, "Train Params:", trainparams)

    args = TrainingArguments(
        output_dir="./NLIMODEL",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        dataloader_pin_memory=True,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=logging_steps,
        disable_tqdm=False,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
