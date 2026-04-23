import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset, DataLoader

from transformers import PreTrainedTokenizerFast
from transformers import Trainer, TrainingArguments, AutoTokenizer

from datasets import load_dataset, Dataset

from model import *

import evaluate

mnli = load_dataset("nyu-mll/multi_nli")

df_val = mnli['validation_matched'].to_pandas()
tokenizer = AutoTokenizer.from_pretrained("./MODEL")
    
def compute_metrics(results):
	pred, targ = results
	pred = np.argmax(pred, axis=-1)
	res = {}
	metric = evaluate.load("accuracy")
	res["accuracy"] = metric.compute(predictions=pred, references=targ)["accuracy"]
	metric = evaluate.load("precision")
	res["precision"] = metric.compute(predictions=pred, references=targ, average="macro", zero_division=0)["precision"]
	metric = evaluate.load("recall")
	res["recall"] = metric.compute(predictions=pred, references=targ, average="macro", zero_division=0)["recall"]
	metric = evaluate.load("f1")
	res["f1"] = metric.compute(predictions=pred, references=targ, average="macro")["f1"]
	return res
	
tokenized_val = (df_val["premise"] + " [CLS] " + df_val["hypothesis"]).apply(lambda sample: tokenizes(sample, tokenizer = tokenizer))
val_set = Dataset.from_dict({"input_ids":[t["input_ids"] for t in tokenized_val], "labels":df_val["label"]})

model = NLI.from_pretrained("./MODEL")

allparams = sum(p.numel() for p in model.parameters())
trainparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("All Param:", allparams, "Train Params:", trainparams)
args = TrainingArguments(per_device_train_batch_size=4)

trainer = Trainer(
    model=model,
    args=args,
    eval_dataset=val_set,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

results = trainer.predict(val_set)
print(results.metrics)
