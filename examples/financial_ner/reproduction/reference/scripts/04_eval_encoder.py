"""Stage 04: evaluate encoder checkpoints on all domains with confidence.

For each seed and each eval set (fin_valid, fin_test, finer_ord_test,
tweetner7_test):
  - token softmax -> BIO decode -> predicted spans
  - span confidence = mean / min of first-subword max-prob within the span
  - temperature scaling fit on fin_valid token logits (per seed), applied to all
  - per-entity and per-sentence records dumped for stage 07 analysis

Usage: python 04_eval_encoder.py
"""
import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer
from common import (DATA_PROC, RUNS, RESULTS, load_config, get_device,
                    write_manifest, bio_to_spans)

EVAL_SETS = {
    "fin_valid": "fin_valid.jsonl",
    "fin_test": "fin_test.jsonl",
    "finer_ord_test": "finer_ord_test.jsonl",
    "tweetner7_test": "tweetner7_test.jsonl",
}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


@torch.no_grad()
def predict(model, tok, rows, device, max_length):
    """Returns per-sentence: word-level logits (first subword)."""
    out = []
    for r in rows:
        enc = tok(r["tokens"], is_split_into_words=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        word_ids = tok(r["tokens"], is_split_into_words=True, truncation=True,
                       max_length=max_length).word_ids()
        logits = model(**enc).logits[0].cpu()
        first_idx, seen = [], set()
        for pos, wid in enumerate(word_ids):
            if wid is not None and wid not in seen:
                seen.add(wid)
                first_idx.append(pos)
        n_words = len(seen)
        out.append({"row": r, "logits": logits[first_idx],
                    "n_words": n_words})
    return out


def fit_temperature(logit_batches, label_batches, device="cpu"):
    logits = torch.cat(logit_batches).float()
    labels = torch.cat(label_batches)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss
    opt.step(closure)
    return float(log_t.exp())


def spans_with_conf(logits, temperature, id2label):
    probs = F.softmax(logits / temperature, dim=-1)
    pred_ids = probs.argmax(-1)
    conf_tok = probs.max(-1).values
    tags = [id2label[int(i)] for i in pred_ids]
    spans = bio_to_spans(tags)
    out = []
    for s, e, t in spans:
        cs = conf_tok[s:e]
        out.append({"start": s, "end": e, "type": t,
                    "conf_mean": float(cs.mean()), "conf_min": float(cs.min())})
    return out, tags


def main():
    cfg = load_config()
    enc_cfg = cfg["encoder"]
    device = get_device()
    RESULTS.mkdir(parents=True, exist_ok=True)
    ent_records, sent_records, temps = [], [], {}

    data = {k: read_jsonl(DATA_PROC / v) for k, v in EVAL_SETS.items()}

    for seed in enc_cfg["seeds"]:
        run_dir = RUNS / f"encoder_seed{seed}"
        tok = AutoTokenizer.from_pretrained(run_dir)
        model = AutoModelForTokenClassification.from_pretrained(run_dir).to(device).eval()
        id2label = model.config.id2label
        l2i = model.config.label2id

        preds = {name: predict(model, tok, rows, device, enc_cfg["max_length"])
                 for name, rows in data.items()}

        # temperature from fin_valid word-level logits vs gold labels
        lb, gb = [], []
        for p in preds["fin_valid"]:
            gold = p["row"]["tags"][:p["n_words"]]
            lb.append(p["logits"])
            gb.append(torch.tensor([l2i[t] for t in gold]))
        T = fit_temperature(lb, gb)
        temps[str(seed)] = T
        print(f"seed {seed}: temperature = {T:.3f}", flush=True)

        for name, plist in preds.items():
            for p in plist:
                r = p["row"]
                gold_spans = {(s, e, t) for s, e, t in bio_to_spans(r["tags"][:p["n_words"]])}
                for T_used, tag in ((1.0, "raw"), (T, "temp")):
                    spans, _ = spans_with_conf(p["logits"], T_used, id2label)
                    pred_set = {(s["start"], s["end"], s["type"]) for s in spans}
                    if tag == "raw":
                        n_err = len(pred_set - gold_spans) + len(gold_spans - pred_set)
                        sent_conf = min((s["conf_mean"] for s in spans), default=1.0)
                        sent_records.append({
                            "model": "encoder", "seed": seed, "domain": name,
                            "sent_id": r["id"], "n_gold": len(gold_spans),
                            "n_pred": len(pred_set), "sent_error": int(n_err > 0),
                            "sent_conf_msp": sent_conf,
                        })
                    for s in spans:
                        key = (s["start"], s["end"], s["type"])
                        ent_records.append({
                            "model": "encoder", "seed": seed, "domain": name,
                            "sent_id": r["id"],
                            "text": " ".join(r["tokens"][s["start"]:s["end"]]),
                            "type": s["type"], "calib": tag,
                            "conf_mean": s["conf_mean"], "conf_min": s["conf_min"],
                            "correct": int(key in gold_spans),
                        })
                # gold recall bookkeeping (raw decode): store missed golds count
                spans_raw, _ = spans_with_conf(p["logits"], 1.0, id2label)
                pred_set = {(s["start"], s["end"], s["type"]) for s in spans_raw}
                sent_records[-1]["n_missed"] = len(gold_spans - pred_set)

    with open(RESULTS / "encoder_entities.jsonl", "w") as f:
        for r in ent_records:
            f.write(json.dumps(r) + "\n")
    with open(RESULTS / "encoder_sentences.jsonl", "w") as f:
        for r in sent_records:
            f.write(json.dumps(r) + "\n")
    with open(RESULTS / "encoder_temperatures.json", "w") as f:
        json.dump(temps, f, indent=2)
    write_manifest("04_eval_encoder", RESULTS,
                   inputs=[DATA_PROC / v for v in EVAL_SETS.values()],
                   extra={"temperatures": temps,
                          "n_entity_records": len(ent_records),
                          "n_sentence_records": len(sent_records)})
    print("eval done")


if __name__ == "__main__":
    main()
