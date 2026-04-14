import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertConfig, BertModel, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput


DEFAULT_PAD_TOKEN_ID = 0
DEFAULT_UNK_TOKEN_ID = 1
DEFAULT_CLS_TOKEN_ID = 2
DEFAULT_SEP_TOKEN_ID = 3
DEFAULT_MASK_TOKEN_ID = 4
DEFAULT_MAX_LENGTH = 256
MAX_PARAMETER_BUDGET = 40_000_000


def _normalize_pair_input(examples):
    if isinstance(examples, dict):
        premise = examples.get("premise") or examples.get("text") or examples.get("sentence1")
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
        sep_positions = torch.nonzero(sep_mask[row_idx], as_tuple=False).flatten()
        if sep_positions.numel() == 0:
            continue

        first_sep = int(sep_positions[0].item())
        if first_sep + 1 >= input_ids.size(1):
            continue

        second_segment = input_ids[row_idx, first_sep + 1 :]
        token_type_ids[row_idx, first_sep + 1 :] = second_segment.ne(pad_token_id).long()

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
    input_ids = pad_sequence(input_sequences, batch_first=True, padding_value=pad_token_id)
    attention_mask = input_ids.ne(pad_token_id).long()
    token_type_ids = _build_token_type_ids(input_ids, sep_token_id=sep_token_id, pad_token_id=pad_token_id)

    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }

    if "labels" in batch[0] and batch[0]["labels"] is not None:
        collated["labels"] = torch.tensor([int(item["labels"]) for item in batch], dtype=torch.long)

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
        vocab_size=26_000,
        hidden_size=512,
        nclass=3,
        num_hidden_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=256,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=DEFAULT_PAD_TOKEN_ID,
        unk_token_id=DEFAULT_UNK_TOKEN_ID,
        cls_token_id=DEFAULT_CLS_TOKEN_ID,
        sep_token_id=DEFAULT_SEP_TOKEN_ID,
        mask_token_id=DEFAULT_MASK_TOKEN_ID,
        classifier_hidden_size=1024,
        classifier_dropout=0.1,
        label_smoothing=0.0,
        hidden_act="gelu",
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.nclass = nclass
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.unk_token_id = unk_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
        self.classifier_hidden_size = classifier_hidden_size
        self.classifier_dropout = classifier_dropout
        self.label_smoothing = label_smoothing
        self.hidden_act = hidden_act

    def to_bert_config(self):
        return BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            hidden_act=self.hidden_act,
        )


def estimated_parameter_count(config):
    hidden_size = config.hidden_size
    intermediate_size = config.intermediate_size

    embedding_params = (
        config.vocab_size * hidden_size
        + config.max_position_embeddings * hidden_size
        + config.type_vocab_size * hidden_size
        + (2 * hidden_size)
    )

    per_layer_params = (
        4 * hidden_size * hidden_size
        + (4 * hidden_size)
        + (2 * hidden_size * intermediate_size)
        + intermediate_size
        + hidden_size
        + (4 * hidden_size)
    )

    pooler_params = hidden_size * hidden_size + hidden_size
    classifier_params = (
        (2 * hidden_size) * config.classifier_hidden_size
        + config.classifier_hidden_size
        + config.classifier_hidden_size * config.nclass
        + config.nclass
    )

    return (
        embedding_params
        + config.num_hidden_layers * per_layer_params
        + pooler_params
        + classifier_params
    )


class NLIClassificationHead(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_labels, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(2 * hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(intermediate_size, num_labels)

    def forward(self, cls_repr, mean_repr):
        pooled = torch.cat([cls_repr, mean_repr], dim=-1)
        pooled = self.dropout(pooled)
        pooled = self.dense(pooled)
        pooled = self.activation(pooled)
        pooled = self.dropout(pooled)
        return self.out_proj(pooled)


class NLI(PreTrainedModel):
    config_class = NLIConfig
    base_model_prefix = "bert"

    def __init__(self, config):
        estimated_params = estimated_parameter_count(config)
        if estimated_params >= MAX_PARAMETER_BUDGET:
            raise ValueError(
                f"NLI configuration exceeds the 40M-parameter budget "
                f"({estimated_params:,} parameters). Retrain with a compatible MODEL/ artifact."
            )

        super().__init__(config)
        self.config = config
        self.bert = BertModel(config.to_bert_config(), add_pooling_layer=True)
        self.classifier = NLIClassificationHead(
            hidden_size=config.hidden_size,
            intermediate_size=config.classifier_hidden_size,
            num_labels=config.nclass,
            dropout=config.classifier_dropout,
        )
        self.loss_fct = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.post_init()

        actual_params = sum(parameter.numel() for parameter in self.parameters())
        if actual_params >= MAX_PARAMETER_BUDGET:
            raise ValueError(
                f"Instantiated NLI model exceeds the 40M-parameter budget "
                f"({actual_params:,} parameters)."
            )

    @staticmethod
    def _masked_mean(last_hidden_state, attention_mask, input_ids, cls_token_id, sep_token_id):
        content_mask = (
            attention_mask.bool()
            & input_ids.ne(cls_token_id)
            & input_ids.ne(sep_token_id)
        )
        mask = content_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        masked_hidden = last_hidden_state * mask
        token_count = mask.sum(dim=1).clamp(min=1.0)
        return masked_hidden.sum(dim=1) / token_count

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.bert.set_input_embeddings(value)

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

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            **kwargs,
        )
        last_hidden_state = outputs.last_hidden_state
        cls_repr = last_hidden_state[:, 0]
        mean_repr = self._masked_mean(
            last_hidden_state,
            attention_mask,
            input_ids,
            cls_token_id=self.config.cls_token_id,
            sep_token_id=self.config.sep_token_id,
        )
        logits = self.classifier(cls_repr, mean_repr)

        loss = None
        if labels is not None:
            loss = self.loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
