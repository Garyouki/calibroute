"""Rebuild the historical NER data splits without model or ML dependencies.

Downloads are pinned and checked before conversion. Every normalized data file
must match the corresponding file recovered from the original experiment Git.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require_hash(data: bytes, expected: str, label: str) -> None:
    actual = sha256(data)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {label}: expected {expected}, got {actual}")


def verify_reference(manifest: dict) -> None:
    for entry in manifest["recovered_files"]:
        require_hash((HERE / entry["path"]).read_bytes(), entry["sha256"], entry["path"])


def load_reference(name: str, filename: str):
    scripts = HERE / "reference" / "scripts"
    spec = importlib.util.spec_from_file_location(name, scripts / filename)
    module = importlib.util.module_from_spec(spec)
    # Reference scripts import common, whose optional ML imports are in functions
    # that this data-only entry point never calls.
    sys.path.insert(0, str(scripts))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def build_splits(blobs: dict[str, bytes], cfg: dict) -> dict[str, dict]:
    download = load_reference("ner_download_reference", "01_download_data.py")
    prep = load_reference("ner_preprocess_reference", "02_preprocess.py")
    fin_labels = {v: k for k, v in json.loads(blobs["fin_label"]).items()}
    tweet_labels = {v: k for k, v in json.loads(blobs["tweetner7_label"]).items()}
    raw = {
        f"fin_{split}.jsonl": download.tner_jsonl(blobs[f"fin_{split}"], fin_labels, "fin", split)
        for split in ("train", "valid", "test")
    }
    raw["tweetner7_test.jsonl"] = download.tner_jsonl(
        blobs["tweetner7_test"], tweet_labels, "tweetner7", "test2021"
    )
    raw["finer_ord_test.jsonl"], _ = download.finer_ord_rows(blobs["finer_ord_test"])
    processed = {}
    rng = random.Random(cfg["subsample_seed"])
    cap = cfg["max_test_sentences_per_domain"]
    # Preserve the historical order: a shared RNG is consumed across splits.
    for name in ("fin_train", "fin_valid", "fin_test", "finer_ord_test", "tweetner7_test"):
        is_tweet = name == "tweetner7_test"
        type_map = cfg["tweetner7_keep_types"] if is_tweet else {
            "PER": "PER", "ORG": "ORG", "LOC": "LOC"
        }
        rows, _ = prep.harmonize(raw[f"{name}.jsonl"], type_map, clean_tokens=is_tweet)
        if name.endswith("test") and len(rows) > cap:
            rows = rng.sample(rows, cap)
            rows.sort(key=lambda row: int(row["id"].rsplit("-", 1)[1]))
        processed[f"{name}.jsonl"] = rows
    return {"raw": raw, "processed": processed}


def encode_rows(rows: list[dict]) -> bytes:
    # Mac source files use LF; explicit bytes preserve their hashes on Windows.
    return "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")


def prepare(output: Path, *, offline: bool = False) -> dict:
    manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    verify_reference(manifest)
    cache = output / "downloads"
    cache.mkdir(parents=True, exist_ok=True)
    blobs = {}
    for entry in manifest["downloads"]:
        path = cache / entry["key"]
        if path.exists():
            data = path.read_bytes()
        elif offline:
            raise FileNotFoundError(f"Offline cache missing: {path}")
        else:
            with urllib.request.urlopen(entry["url"], timeout=60) as response:
                data = response.read()
        require_hash(data, entry["sha256"], entry["key"])
        if not path.exists():
            path.write_bytes(data)
        blobs[entry["key"]] = data

    splits = build_splits(blobs, manifest["preprocess"])
    verified = []
    payloads = []
    for directory, files in splits.items():
        for name, rows in files.items():
            expected = manifest["expected_data"][directory][name]
            data = encode_rows(rows)
            require_hash(data, expected["sha256"], f"{directory}/{name}")
            if len(rows) != expected["rows"]:
                raise ValueError(f"Row count mismatch for {directory}/{name}")
            payloads.append((output / directory / name, data))
            verified.append({"path": f"{directory}/{name}", "rows": len(rows), "sha256": sha256(data)})
    # Only write normalized splits after every historical hash has matched.
    for path, data in payloads:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    report = {
        "source_commit": manifest["source_commit"],
        "source_manifest_sha256": sha256((HERE / "source_manifest.json").read_bytes()),
        "runner_sha256": sha256(Path(__file__).read_bytes()),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "status": "all_historical_data_hashes_match",
        "verified_files": verified,
        "scope": "Data acquisition and preprocessing only; no training or prediction reproduction.",
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for downloads and prepared data")
    parser.add_argument("--offline", action="store_true", help="Use the previously verified download cache")
    args = parser.parse_args()
    try:
        report = prepare(args.output, offline=args.offline)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    print(f"Verified {len(report['verified_files'])} historical data files. See {args.output / 'verification.json'}")


if __name__ == "__main__":
    main()
