import evaluate
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, Trainer, TrainingArguments

from model import NLI, collate_fn


MODEL_NAME = "albert-large-v2"
MODEL_DIR = "./MODEL"
MAX_LENGTH = 256


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
    print("Loading MNLI datasets...")
    mnli = load_dataset("nyu-mll/multi_nli")

    train_ds = mnli["train"].filter(lambda x: x["label"] in (0, 1, 2))
    val_ds = mnli["validation_matched"].filter(
        lambda x: x["label"] in (0, 1, 2))

    print("Loading ALBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tqdm.write("Tokenizing datasets (tqdm progress from datasets.map)...")

    def preprocess(batch):
        encoded = tokenizer(
            batch["premise"],
            text_pair=batch["hypothesis"],
            truncation="longest_first",
            max_length=MAX_LENGTH,
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
        desc="Tokenizing train",
        remove_columns=train_ds.column_names,
    )
    val_set = val_ds.map(
        preprocess,
        batched=True,
        desc="Tokenizing validation",
        remove_columns=val_ds.column_names,
    )

    model = NLI.from_albert_pretrained(
        MODEL_NAME, num_labels=3, classifier_dropout=0.1)

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
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        disable_tqdm=False,
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

    trainer.model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)


if __name__ == "__main__":
    main()
