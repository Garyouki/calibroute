"""Stage 01: download the three source datasets (raw files from HuggingFace
repos — the tner repos use legacy loader scripts unsupported by datasets>=3)
and convert to a common raw JSONL: {"id", "tokens", "tags"} with string BIO tags.

Domains (per RESEARCH_PLAN.md §5.3):
  - tner/fin               SEC filings (train domain)   [Salinas Alvarado 2015]
  - gtfintechlab/finer-ord financial news (near shift)  [Shah 2023]
  - tner/tweetner7         tweets (far shift)           [Ushio 2022]
"""
import csv
import io
import json
import urllib.request
from common import DATA_RAW, write_manifest

HF = "https://huggingface.co/datasets"

SOURCES = {
    "fin_label": f"{HF}/tner/fin/resolve/main/dataset/label.json",
    "fin_train": f"{HF}/tner/fin/resolve/main/dataset/train.json",
    "fin_valid": f"{HF}/tner/fin/resolve/main/dataset/valid.json",
    "fin_test": f"{HF}/tner/fin/resolve/main/dataset/test.json",
    "tweetner7_label": f"{HF}/tner/tweetner7/resolve/main/dataset/label.json",
    "tweetner7_test": f"{HF}/tner/tweetner7/resolve/main/dataset/2021.test.json",
    "finer_ord_test": f"{HF}/gtfintechlab/finer-ord/resolve/main/test.csv",
}


def fetch(url):
    with urllib.request.urlopen(url) as r:
        return r.read()


def tner_jsonl(raw_bytes, id2label, name, split):
    rows = []
    for i, line in enumerate(raw_bytes.decode("utf-8").strip().split("\n")):
        ex = json.loads(line)
        rows.append({
            "id": f"{name}-{split}-{i}",
            "tokens": ex["tokens"],
            "tags": [id2label[t] for t in ex["tags"]],
        })
    return rows


FINER_ORD_MAP = {"0": "O", "1": "B-PER", "2": "I-PER", "3": "B-LOC",
                 "4": "I-LOC", "5": "B-ORG", "6": "I-ORG"}


def finer_ord_rows(raw_bytes):
    """token-per-row CSV -> sentence rows grouped by (doc_idx, sent_idx)."""
    reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8")))
    fields = reader.fieldnames
    tok_f = next(c for c in ("gold_token", "token", "tokens") if c in fields)
    tag_f = next(c for c in ("gold_label", "gold_labels", "ner_tags", "tags", "label")
                 if c in fields)
    doc_f = "doc_idx" if "doc_idx" in fields else None
    sent_f = "sent_idx" if "sent_idx" in fields else None
    grouped = {}
    order = []
    for r in reader:
        tok = r[tok_f]
        if tok is None or tok == "":
            continue
        key = (r.get(doc_f, "0"), r.get(sent_f, "0"))
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        raw_tag = str(r[tag_f]).strip()
        tag = FINER_ORD_MAP.get(raw_tag.split(".")[0], raw_tag if raw_tag.startswith(("B-", "I-", "O")) else "O")
        grouped[key].append((tok, tag))
    rows = []
    for i, key in enumerate(order):
        toks, tags = zip(*grouped[key])
        rows.append({"id": f"finer_ord-test-{i}", "tokens": list(toks), "tags": list(tags)})
    return rows, fields


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    written, extra = [], {"sources": SOURCES}

    blobs = {k: fetch(u) for k, u in SOURCES.items()}

    fin_labels = json.loads(blobs["fin_label"])          # {"O": 0, ...}
    fin_id2l = {v: k for k, v in fin_labels.items()}
    tw_labels = json.loads(blobs["tweetner7_label"])
    tw_id2l = {v: k for k, v in tw_labels.items()}
    extra["fin_labels"] = fin_labels
    extra["tweetner7_labels"] = tw_labels

    outputs = {}
    for split in ("train", "valid", "test"):
        outputs[f"fin_{split}.jsonl"] = tner_jsonl(blobs[f"fin_{split}"], fin_id2l, "fin", split)
    outputs["tweetner7_test.jsonl"] = tner_jsonl(blobs["tweetner7_test"], tw_id2l, "tweetner7", "test2021")
    ford_rows, ford_fields = finer_ord_rows(blobs["finer_ord_test"])
    extra["finer_ord_csv_fields"] = ford_fields
    outputs["finer_ord_test.jsonl"] = ford_rows

    for fname, rows in outputs.items():
        path = DATA_RAW / fname
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        written.append(path)
        print(f"wrote {path} ({len(rows)} sentences)")

    write_manifest("01_download", DATA_RAW, inputs=written, extra=extra)


if __name__ == "__main__":
    main()
