import argparse
import json
import math
from pathlib import Path


LABEL_TO_ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "e": 0,
    "n": 1,
    "c": 2,
}

HF_DOWNLOAD_SPECS = {
    "mnli_train": [("nyu-mll/multi_nli", None, "train", "mnli_train")],
    "mnli_validation_matched": [("nyu-mll/multi_nli", None, "validation_matched", "mnli_validation_matched")],
    "mnli_validation_mismatched": [("nyu-mll/multi_nli", None, "validation_mismatched", "mnli_validation_mismatched")],
    "snli_train": [("stanfordnlp/snli", None, "train", "snli_train")],
    "snli_validation": [("stanfordnlp/snli", None, "validation", "snli_validation")],
    "snli_test": [("stanfordnlp/snli", None, "test", "snli_test")],
    "anli_train": [
        ("facebook/anli", None, "train_r1", "anli_train_r1"),
        ("facebook/anli", None, "train_r2", "anli_train_r2"),
        ("facebook/anli", None, "train_r3", "anli_train_r3"),
    ],
    "anli_train_r1": [("facebook/anli", None, "train_r1", "anli_train_r1")],
    "anli_train_r2": [("facebook/anli", None, "train_r2", "anli_train_r2")],
    "anli_train_r3": [("facebook/anli", None, "train_r3", "anli_train_r3")],
    "anli_dev_r1": [("facebook/anli", None, "dev_r1", "anli_dev_r1")],
    "anli_dev_r2": [("facebook/anli", None, "dev_r2", "anli_dev_r2")],
    "anli_dev_r3": [("facebook/anli", None, "dev_r3", "anli_dev_r3")],
    "anli_test_r1": [("facebook/anli", None, "test_r1", "anli_test_r1")],
    "anli_test_r2": [("facebook/anli", None, "test_r2", "anli_test_r2")],
    "anli_test_r3": [("facebook/anli", None, "test_r3", "anli_test_r3")],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize local English NLI files into data/processed JSONL. "
        "If a raw file is missing, known splits can be downloaded automatically from Hugging Face."
    )
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help=(
            "Mapping in the form split_name=path/to/raw.jsonl. "
            "If the file is missing and split_name is known, the script will try to download it."
        ),
    )
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--no-download-missing", action="store_true")
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


def read_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    if path.suffix.lower() == ".jsonl":
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

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for candidate_key in ("data", "examples", "rows"):
                if isinstance(payload.get(candidate_key), list):
                    return payload[candidate_key]
        raise ValueError(f"Unsupported JSON structure in {path}. Expected a list of rows.")

    raise ValueError(f"Unsupported raw data format for {path}. Use .jsonl or .json.")


def normalize_example(row, split_name):
    premise = row.get("premise") or row.get("sentence1")
    hypothesis = row.get("hypothesis") or row.get("sentence2")
    label = normalize_label(row.get("label"))

    if not isinstance(premise, str) or not isinstance(hypothesis, str):
        return None

    premise = premise.strip()
    hypothesis = hypothesis.strip()
    if not premise or not hypothesis or label is None:
        return None

    source = row.get("source")
    if not isinstance(source, str) or not source.strip():
        source = split_name

    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "source": source.strip().lower(),
    }


def parse_spec(spec):
    if "=" not in spec:
        raise ValueError(
            f"Invalid --spec value '{spec}'. Expected split_name=path/to/raw.jsonl"
        )
    split_name, raw_path = spec.split("=", 1)
    split_name = split_name.strip()
    raw_path = raw_path.strip()
    if not split_name or not raw_path:
        raise ValueError(
            f"Invalid --spec value '{spec}'. Expected split_name=path/to/raw.jsonl"
        )
    return split_name, raw_path


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dataset_module():
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Automatic download requires the 'datasets' package. "
            "Install it first with `pip install datasets` or provide local raw files."
        ) from error
    return load_dataset


def download_rows_for_split(split_name, raw_path):
    download_specs = HF_DOWNLOAD_SPECS.get(split_name)
    if not download_specs:
        raise FileNotFoundError(
            f"Raw data file not found: {raw_path}. "
            f"Split '{split_name}' is not in the automatic download map."
        )

    load_dataset = load_dataset_module()
    downloaded_rows = []
    for dataset_name, subset_name, dataset_split, source_name in download_specs:
        kwargs = {"path": dataset_name, "split": dataset_split}
        if subset_name is not None:
            kwargs["name"] = subset_name
        dataset = load_dataset(**kwargs)
        for row in dataset:
            downloaded_rows.append(
                {
                    "premise": row.get("premise"),
                    "hypothesis": row.get("hypothesis"),
                    "label": row.get("label"),
                    "source": source_name,
                }
            )
    return downloaded_rows


def load_or_download_rows(split_name, raw_path, allow_download):
    raw_path = Path(raw_path)
    if raw_path.exists():
        return read_rows(raw_path), "local"

    if not allow_download:
        raise FileNotFoundError(f"Raw data file not found: {raw_path}")

    rows = download_rows_for_split(split_name, raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(raw_path, rows)
    return rows, "downloaded"


def default_specs(raw_dir):
    raw_dir = Path(raw_dir)
    return [
        f"mnli_train={raw_dir / 'mnli_train.jsonl'}",
        f"mnli_validation_mismatched={raw_dir / 'mnli_validation_mismatched.jsonl'}",
        f"snli_train={raw_dir / 'snli_train.jsonl'}",
        f"snli_validation={raw_dir / 'snli_validation.jsonl'}",
        f"snli_test={raw_dir / 'snli_test.jsonl'}",
        f"anli_train={raw_dir / 'anli_train.jsonl'}",
    ]


def main():
    args = parse_args()
    if not args.spec:
        args.spec = default_specs(args.raw_dir)

    output_dir = Path(args.output_dir)
    manifest = {}

    for item in args.spec:
        split_name, raw_path = parse_spec(item)
        rows, source_kind = load_or_download_rows(
            split_name=split_name,
            raw_path=raw_path,
            allow_download=not args.no_download_missing,
        )
        normalized = []
        seen = set()

        for row in rows:
            example = normalize_example(row, split_name=split_name)
            if example is None:
                continue

            dedupe_key = (example["premise"], example["hypothesis"], example["label"])
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(example)

        output_path = output_dir / f"{split_name}.jsonl"
        write_jsonl(output_path, normalized)
        manifest[split_name] = {
            "raw_path": str(raw_path),
            "raw_source": source_kind,
            "output_path": str(output_path),
            "num_examples": len(normalized),
        }
        print(f"{split_name}: wrote {len(normalized):,} examples to {output_path} ({source_kind})")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
