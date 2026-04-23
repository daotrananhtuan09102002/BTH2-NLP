import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AlbertConfig, AlbertModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput


DEFAULT_PAD_TOKEN_ID = 0
DEFAULT_UNK_TOKEN_ID = 1
DEFAULT_CLS_TOKEN_ID = 2
DEFAULT_SEP_TOKEN_ID = 3
DEFAULT_MASK_TOKEN_ID = 4
DEFAULT_MAX_LENGTH = 256


def _normalize_pair_input(examples):
    if isinstance(examples, dict):
        premise = examples.get("premise") or examples.get(
            "text") or examples.get("sentence1")
        hypothesis = examples.get("hypothesis") or examples.get("sentence2")
        return premise, hypothesis

    if isinstance(examples, (list, tuple)) and len(examples) == 2:
        return examples[0], examples[1]

    if isinstance(examples, str):
        if " [CLS] " in examples:
            premise, hypothesis = examples.split(" [CLS] ", 1)
            return premise.strip(), hypothesis.strip()
        return examples, None

    raise TypeError(
        "Unsupported input format for tokenizes(). Use a string, a 2-item tuple/list, "
        "or a dict with premise/hypothesis fields."
    )


def _build_token_type_ids(input_ids, sep_token_id, pad_token_id):
    token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    sep_mask = input_ids.eq(sep_token_id)

    for row_idx in range(input_ids.size(0)):
        sep_positions = torch.nonzero(
            sep_mask[row_idx], as_tuple=False).flatten()
        if sep_positions.numel() == 0:
            continue

        first_sep = int(sep_positions[0].item())
        if first_sep + 1 >= input_ids.size(1):
            continue

        second_segment = input_ids[row_idx, first_sep + 1:]
        token_type_ids[row_idx, first_sep +
                       1:] = second_segment.ne(pad_token_id).long()

    return token_type_ids


def collate_fn(batch):
    if not batch:
        raise ValueError("collate_fn() received an empty batch.")

    pad_token_id = int(batch[0].get("pad_token_id", DEFAULT_PAD_TOKEN_ID))
    sep_token_id = int(batch[0].get("sep_token_id", DEFAULT_SEP_TOKEN_ID))
    input_sequences = [
        torch.tensor(item["input_ids"], dtype=torch.long)
        if not isinstance(item["input_ids"], torch.Tensor)
        else item["input_ids"].long()
        for item in batch
    ]
    input_ids = pad_sequence(
        input_sequences, batch_first=True, padding_value=pad_token_id)
    attention_mask = input_ids.ne(pad_token_id).long()
    token_type_ids = _build_token_type_ids(
        input_ids, sep_token_id=sep_token_id, pad_token_id=pad_token_id)

    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    if "labels" in batch[0] and batch[0]["labels"] is not None:
        collated["labels"] = torch.tensor(
            [int(item["labels"]) for item in batch], dtype=torch.long)

    return collated


def tokenizes(examples, tokenizer, max_length=DEFAULT_MAX_LENGTH):
    premise, hypothesis = _normalize_pair_input(examples)
    kwargs = {
        "truncation": "longest_first",
        "max_length": max_length,
        "padding": "max_length",
    }

    if hypothesis is None:
        return tokenizer(premise, **kwargs)
    return tokenizer(premise, text_pair=hypothesis, **kwargs)


class NLIConfig(PretrainedConfig):
    model_type = "NLI"

    def __init__(
        self,
        base_model_name="albert-large-v2",
        albert_config=None,
        nclass=3,
        hidden_size=1024,
        hidden_dropout_prob=0.1,
        pad_token_id=DEFAULT_PAD_TOKEN_ID,
        unk_token_id=DEFAULT_UNK_TOKEN_ID,
        cls_token_id=DEFAULT_CLS_TOKEN_ID,
        sep_token_id=DEFAULT_SEP_TOKEN_ID,
        mask_token_id=DEFAULT_MASK_TOKEN_ID,
        classifier_dropout=0.1,
        label_smoothing=0.0,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.base_model_name = base_model_name
        self.albert_config = albert_config
        self.hidden_size = hidden_size
        self.nclass = nclass
        self.hidden_dropout_prob = hidden_dropout_prob
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
        self.classifier_dropout = classifier_dropout
        self.label_smoothing = label_smoothing

    def to_albert_config(self):
        if self.albert_config is not None:
            return AlbertConfig.from_dict(self.albert_config)

        return AlbertConfig(
            hidden_size=self.hidden_size,
            classifier_dropout_prob=self.classifier_dropout,
            hidden_dropout_prob=self.hidden_dropout_prob,
            pad_token_id=self.pad_token_id,
        )


class NLIClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, pooled_output):
        pooled = pooled_output
        pooled = self.dropout(pooled)
        pooled = self.dense(pooled)
        pooled = self.activation(pooled)
        pooled = self.dropout(pooled)
        return self.out_proj(pooled)


class NLI(PreTrainedModel):
    config_class = NLIConfig
    base_model_prefix = "albert"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        albert_config = config.to_albert_config()
        # Keep special token IDs consistent with tokenizer artifacts saved in MODEL/.
        albert_config.pad_token_id = config.pad_token_id
        albert_config.bos_token_id = config.cls_token_id
        albert_config.eos_token_id = config.sep_token_id

        if config.hidden_size is None:
            config.hidden_size = int(albert_config.hidden_size)
        else:
            config.hidden_size = int(albert_config.hidden_size)

        self.albert = AlbertModel(albert_config, add_pooling_layer=True)
        self.classifier = NLIClassificationHead(
            hidden_size=albert_config.hidden_size,
            num_labels=config.nclass,
            dropout=config.classifier_dropout,
        )
        self.loss_fct = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing)
        self.post_init()

    def get_input_embeddings(self):
        return self.albert.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.albert.set_input_embeddings(value)

    @classmethod
    def from_albert_pretrained(cls, model_name_or_path, num_labels=3, classifier_dropout=0.1):
        albert_config = AlbertConfig.from_pretrained(model_name_or_path)
        config = NLIConfig(
            base_model_name=model_name_or_path,
            albert_config=albert_config.to_dict(),
            hidden_size=albert_config.hidden_size,
            nclass=num_labels,
            hidden_dropout_prob=albert_config.hidden_dropout_prob,
            classifier_dropout=classifier_dropout,
            pad_token_id=albert_config.pad_token_id,
            cls_token_id=getattr(
                albert_config, "bos_token_id", DEFAULT_CLS_TOKEN_ID),
            sep_token_id=getattr(
                albert_config, "eos_token_id", DEFAULT_SEP_TOKEN_ID),
        )
        model = cls(config)
        backbone = AlbertModel.from_pretrained(
            model_name_or_path, config=albert_config)
        model.albert.load_state_dict(backbone.state_dict(), strict=False)
        return model

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id).long()

        if token_type_ids is None:
            token_type_ids = _build_token_type_ids(
                input_ids,
                sep_token_id=self.config.sep_token_id,
                pad_token_id=self.config.pad_token_id,
            )

        # Newer Trainer versions may pass bookkeeping kwargs (e.g. num_items_in_batch)
        # that are not valid AlbertModel.forward arguments.
        kwargs.pop("num_items_in_batch", None)

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            **kwargs,
        )
        pooled_output = outputs.pooler_output
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
