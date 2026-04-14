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


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize local English NLI files into data/processed JSONL.")
    parser.add_argument(
        "--spec",
        action="append",
        default=[],
        help=(
            "Mapping in the form split_name=path/to/raw.jsonl. "
            "Repeat this argument for each split you want to create."
        ),
    )
    parser.add_argument("--output-dir", default="data/processed")
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


def main():
    args = parse_args()
    if not args.spec:
        raise ValueError(
            "No input specs provided. Example: "
            "--spec mnli_train=data/raw/mnli_train.jsonl --spec anli_train=data/raw/anli_train.jsonl"
        )

    output_dir = Path(args.output_dir)
    manifest = {}

    for item in args.spec:
        split_name, raw_path = parse_spec(item)
        rows = read_rows(raw_path)
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
            "raw_path": raw_path,
            "output_path": str(output_path),
            "num_examples": len(normalized),
        }
        print(f"{split_name}: wrote {len(normalized):,} examples to {output_path}")

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
