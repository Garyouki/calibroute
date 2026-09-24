"""Stage 03: fine-tune the encoder baseline (BERT token classifier) on FIN.

Usage: python 03_train_encoder.py --seed 42
Writes runs/encoder_seed{SEED}/ with model, tokenizer, loss log, manifest.
"""
import argparse
import json
import math
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup
from common import DATA_PROC, RUNS, load_config, set_seed, get_device, write_manifest

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
L2I = {l: i for i, l in enumerate(LABELS)}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def encode(rows, tokenizer, max_length):
    feats = []
    for r in rows:
        enc = tokenizer(r["tokens"], is_split_into_words=True, truncation=True,
                        max_length=max_length)
        word_ids = enc.word_ids()
        labels, prev = [], None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev:
                labels.append(L2I[r["tags"][wid]])
            else:
                labels.append(-100)  # only first subword carries the label
            prev = wid
        feats.append({"input_ids": enc["input_ids"],
                      "attention_mask": enc["attention_mask"],
                      "labels": labels})
    return feats


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        d = n - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_id] * d)
        out["attention_mask"].append(b["attention_mask"] + [0] * d)
        out["labels"].append(b["labels"] + [-100] * d)
    return {k: torch.tensor(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config()["encoder"]
    set_seed(args.seed)
    device = get_device()

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["model_name"], num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)}, label2id=L2I).to(device)

    train = encode(read_jsonl(DATA_PROC / "fin_train.jsonl"), tok, cfg["max_length"])
    g = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(train, batch_size=cfg["batch_size"], shuffle=True, generator=g,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    steps = len(dl) * cfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = get_linear_schedule_with_warmup(opt, int(steps * cfg["warmup_ratio"]), steps)

    out_dir = RUNS / f"encoder_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    model.train()
    for ep in range(cfg["epochs"]):
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            if i % 20 == 0:
                log.append({"epoch": ep, "step": i, "loss": float(loss)})
                print(f"ep{ep} step{i} loss {float(loss):.4f}", flush=True)

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    write_manifest("03_train_encoder", out_dir, args=vars(args) | cfg,
                   inputs=[DATA_PROC / "fin_train.jsonl"],
                   extra={"train_minutes": (time.time() - t0) / 60,
                          "final_loss": log[-1]["loss"] if log else None})
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
