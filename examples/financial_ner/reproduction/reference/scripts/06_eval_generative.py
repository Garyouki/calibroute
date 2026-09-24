"""Stage 06: evaluate the generative model with confidence signals.

Per sentence: one greedy decode with token logprobs + K sampled decodes.
Signals per predicted entity:
  - conf_seq    : length-normalized sequence probability of the whole output
  - conf_tokens : mean token probability of the whole output
  - conf_span   : exp(mean logprob) over the tokens of the entity text value
  - conf_type   : exp(mean logprob) over the tokens of the entity type value
  - conf_sc     : self-consistency vote share over K sampled decodes
Also: invalid-JSON flag, hallucination flag (span not in source), and
per-sentence tp/fp/fn for selective-F1 analysis.

Usage: python 06_eval_generative.py --seed 42 [--sets fin_valid,...]
"""
import argparse
import json
import math
import re
import time
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import (DATA_PROC, RUNS, RESULTS, load_config, set_seed, get_device,
                    write_manifest, bio_to_spans, match_entities, norm_text)
from importlib import import_module
_gen_train = import_module("05_train_generative")
SYSTEM, make_messages = _gen_train.SYSTEM, _gen_train.make_messages

EVAL_SETS = ["fin_valid", "fin_test", "finer_ord_test", "tweetner7_test"]
ENT_RE = re.compile(r'\{\s*"text"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"type"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}')
VALID_TYPES = {"PER", "ORG", "LOC"}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def parse_entities(text):
    """Parse model output. Returns (entities, invalid_json_flag).
    entities: list of dicts with text/type and char span of match groups."""
    invalid = 0
    try:
        start = text.index("{")
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    json.loads(text[start:i + 1])
                    break
        else:
            raise ValueError
    except (ValueError, json.JSONDecodeError):
        invalid = 1
    ents = []
    for m in ENT_RE.finditer(text):
        raw_t, raw_y = m.group(1), m.group(2)
        try:
            t = json.loads(f'"{raw_t}"')
        except json.JSONDecodeError:
            t = raw_t
        ents.append({"text": t, "type": raw_y,
                     "text_span": m.span(1), "type_span": m.span(2)})
    return ents, invalid


def token_char_offsets(tok, ids):
    """Char span of each generated token in the decoded string."""
    offsets, prev = [], ""
    for k in range(1, len(ids) + 1):
        cur = tok.decode(ids[:k], skip_special_tokens=True)
        offsets.append((len(prev), len(cur)))
        prev = cur
    return prev, offsets


def mean_logprob_over(char_lo, char_hi, offsets, logprobs):
    vals = [lp for (a, b), lp in zip(offsets, logprobs) if b > char_lo and a < char_hi]
    return sum(vals) / len(vals) if vals else float("-inf")


@torch.no_grad()
def generate_batch(model, tok, prompts, device, max_new, do_sample, temperature, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    out = model.generate(
        **enc, max_new_tokens=max_new, do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=0.95 if do_sample else None,
        output_scores=not do_sample, return_dict_in_generate=True,
        pad_token_id=tok.pad_token_id)
    seqs = out.sequences[:, enc["input_ids"].shape[1]:].cpu()
    results = []
    for b in range(seqs.shape[0]):
        ids = [int(t) for t in seqs[b] if int(t) != tok.pad_token_id]
        # strip trailing eos/special
        lps = None
        if not do_sample:
            lps = []
            for step, tid in enumerate(seqs[b]):
                tid = int(tid)
                if tid == tok.pad_token_id:
                    break
                logit = out.scores[step][b].float()
                lps.append(float(torch.log_softmax(logit, -1)[tid]))
            lps = lps[:len(ids)]
        results.append((ids, lps))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--sets", default=",".join(EVAL_SETS))
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config()["generative"]
    set_seed(args.seed)
    device = get_device()
    run_dir = RUNS / f"generative_seed{args.seed}"

    tok = AutoTokenizer.from_pretrained(run_dir)
    base = AutoModelForCausalLM.from_pretrained(cfg["model_name"], dtype=torch.float16)
    model = PeftModel.from_pretrained(base, run_dir).to(device).eval()

    K, max_new = cfg["self_consistency_k"], cfg["max_new_tokens"]
    t0 = time.time()
    all_counts = {}

    for set_name in args.sets.split(","):
        # per-set output files -> progress survives interruption; skip done sets
        ent_path = RESULTS / f"generative_entities_seed{args.seed}_{set_name}.jsonl"
        sent_path = RESULTS / f"generative_sentences_seed{args.seed}_{set_name}.jsonl"
        if ent_path.exists() and sent_path.exists():
            print(f"[{set_name}] already done, skipping", flush=True)
            continue
        ent_records, sent_records = [], []
        rows = read_jsonl(DATA_PROC / f"{set_name}.jsonl")
        for lo in range(0, len(rows), args.batch_size):
            chunk = rows[lo:lo + args.batch_size]
            prompts = [tok.apply_chat_template(
                make_messages(" ".join(r["tokens"])), tokenize=False,
                add_generation_prompt=True) for r in chunk]

            greedy = generate_batch(model, tok, prompts, device, max_new,
                                    False, None)
            samples = [generate_batch(model, tok, prompts, device, max_new,
                                      True, cfg["sc_temperature"],
                                      seed=args.seed * 1000 + k)
                       for k in range(K)]

            for j, r in enumerate(chunk):
                ids, lps = greedy[j]
                text, offsets = token_char_offsets(tok, ids)
                ents, invalid = parse_entities(text)
                conf_seq = math.exp(sum(lps) / len(lps)) if lps else 0.0
                conf_tokens = (sum(math.exp(x) for x in lps) / len(lps)) if lps else 0.0

                sample_ent_sets = []
                for k in range(K):
                    s_ids, _ = samples[k][j]
                    s_text = tok.decode(s_ids, skip_special_tokens=True)
                    s_ents, _ = parse_entities(s_text)
                    sample_ent_sets.append({(norm_text(e["text"]), e["type"])
                                            for e in s_ents})

                gold = [(" ".join(r["tokens"][s:e]), t)
                        for s, e, t in bio_to_spans(r["tags"])]
                pred = [(e["text"], e["type"]) for e in ents
                        if e["type"] in VALID_TYPES]
                pred_ents = [e for e in ents if e["type"] in VALID_TYPES]
                tp, fp, fn, matched = match_entities(gold, pred)
                src_norm = norm_text(" ".join(r["tokens"]))

                ent_confs = []
                for i, e in enumerate(pred_ents):
                    span_lp = mean_logprob_over(*e["text_span"], offsets, lps)
                    type_lp = mean_logprob_over(*e["type_span"], offsets, lps)
                    conf_span = math.exp(span_lp) if span_lp > -1e8 else 0.0
                    conf_type = math.exp(type_lp) if type_lp > -1e8 else 0.0
                    votes = sum((norm_text(e["text"]), e["type"]) in s
                                for s in sample_ent_sets) / K
                    halluc = int(norm_text(e["text"]) not in src_norm)
                    rec = {"model": "generative", "seed": args.seed,
                           "domain": set_name, "sent_id": r["id"],
                           "text": e["text"], "type": e["type"],
                           "conf_seq": conf_seq, "conf_tokens": conf_tokens,
                           "conf_span": conf_span, "conf_type": conf_type,
                           "conf_sc": votes, "hallucinated": halluc,
                           "correct": int(i in matched)}
                    ent_records.append(rec)
                    ent_confs.append(rec)

                sent_records.append({
                    "model": "generative", "seed": args.seed, "domain": set_name,
                    "sent_id": r["id"], "invalid_json": invalid,
                    "n_gold": len(gold), "n_pred": len(pred),
                    "tp": tp, "fp": fp, "fn": fn,
                    "sent_error": int(fp + fn > 0),
                    "conf_seq": conf_seq, "conf_tokens": conf_tokens,
                    "conf_min_span": min((e["conf_span"] for e in ent_confs), default=1.0),
                    "conf_min_sc": min((e["conf_sc"] for e in ent_confs), default=1.0),
                })
            done = lo + len(chunk)
            if (lo // args.batch_size) % 5 == 0:
                el = time.time() - t0
                print(f"[{set_name}] {done}/{len(rows)} elapsed {el/60:.1f}m", flush=True)

        with open(ent_path, "w") as f:
            for r in ent_records:
                f.write(json.dumps(r) + "\n")
        with open(sent_path, "w") as f:
            for r in sent_records:
                f.write(json.dumps(r) + "\n")
        all_counts[set_name] = {"entities": len(ent_records),
                                "sentences": len(sent_records)}
        print(f"[{set_name}] written ({len(sent_records)} sentences)", flush=True)

    write_manifest(f"06_eval_generative_seed{args.seed}", RESULTS,
                   args=vars(args),
                   inputs=[DATA_PROC / f"{s}.jsonl" for s in args.sets.split(",")],
                   extra={"minutes": (time.time() - t0) / 60, "counts": all_counts})
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
