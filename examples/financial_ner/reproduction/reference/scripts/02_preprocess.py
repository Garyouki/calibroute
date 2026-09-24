"""Stage 02: harmonize the three datasets to a shared PER/ORG/LOC schema and
build the final splits.

Decisions (documented in the paper):
  - FIN sentences containing MISC entities are dropped (schema cleanliness)
  - TweetNER7 sentences are kept only if all their entities are within
    {corporation->ORG, person->PER, location->LOC}; other tweets are dropped
    so the gold labels never silently relabel an entity as O.
  - Tweet special tokens are cleaned: {{URL}} -> URL, {{USERNAME}} -> @user,
    {@X@} -> X (keeps the entity surface form).
  - OOD test sets subsampled to <=300 sentences with a fixed seed.
"""
import json
import random
from collections import Counter
from common import DATA_RAW, DATA_PROC, load_config, write_manifest, bio_to_spans


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path} ({len(rows)} sentences)")


def clean_tweet_token(tok):
    if tok == "{{URL}}":
        return "URL"
    if tok == "{{USERNAME}}":
        return "@user"
    if tok.startswith("{@") and tok.endswith("@}"):
        return tok[2:-2]
    return tok


def harmonize(rows, type_map, drop_unmapped_sentence=True, clean_tokens=False):
    """type_map: raw entity type -> harmonized type (or None to drop sentence)."""
    kept, dropped = [], 0
    for r in rows:
        spans = bio_to_spans(r["tags"])
        types = {t for _, _, t in spans}
        unmapped = {t for t in types if t not in type_map}
        if unmapped and drop_unmapped_sentence:
            dropped += 1
            continue
        new_tags = []
        for tag in r["tags"]:
            if tag == "O":
                new_tags.append("O")
            else:
                prefix, typ = tag.split("-", 1)
                m = type_map.get(typ)
                new_tags.append(f"{prefix}-{m}" if m else "O")
        toks = [clean_tweet_token(t) for t in r["tokens"]] if clean_tokens else r["tokens"]
        kept.append({"id": r["id"], "tokens": toks, "tags": new_tags})
    return kept, dropped


def stats(rows):
    n_ent = Counter()
    for r in rows:
        for _, _, t in bio_to_spans(r["tags"]):
            n_ent[t] += 1
    return {"sentences": len(rows), "entities": sum(n_ent.values()), "by_type": dict(n_ent)}


def main():
    cfg = load_config()
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg["preprocess"]["subsample_seed"])
    cap = cfg["preprocess"]["max_test_sentences_per_domain"]
    report = {}

    fin_map = {"PER": "PER", "ORG": "ORG", "LOC": "LOC"}  # MISC -> drop sentence
    for split in ("train", "valid", "test"):
        rows = read_jsonl(DATA_RAW / f"fin_{split}.jsonl")
        kept, dropped = harmonize(rows, fin_map)
        if split == "test" and len(kept) > cap:
            kept = rng.sample(kept, cap)
            kept.sort(key=lambda r: int(r["id"].rsplit("-", 1)[1]))
        write_jsonl(DATA_PROC / f"fin_{split}.jsonl", kept)
        report[f"fin_{split}"] = {**stats(kept), "dropped_misc_sentences": dropped}

    ford = read_jsonl(DATA_RAW / "finer_ord_test.jsonl")
    kept, dropped = harmonize(ford, {"PER": "PER", "ORG": "ORG", "LOC": "LOC"})
    if len(kept) > cap:
        kept = rng.sample(kept, cap)
        kept.sort(key=lambda r: int(r["id"].rsplit("-", 1)[1]))
    write_jsonl(DATA_PROC / "finer_ord_test.jsonl", kept)
    report["finer_ord_test"] = {**stats(kept), "dropped_sentences": dropped}

    tw_map = {k: v for k, v in cfg["preprocess"]["tweetner7_keep_types"].items()}
    tw = read_jsonl(DATA_RAW / "tweetner7_test.jsonl")
    kept, dropped = harmonize(tw, tw_map, clean_tokens=True)
    if len(kept) > cap:
        kept = rng.sample(kept, cap)
        kept.sort(key=lambda r: int(r["id"].rsplit("-", 1)[1]))
    write_jsonl(DATA_PROC / "tweetner7_test.jsonl", kept)
    report["tweetner7_test"] = {**stats(kept), "dropped_out_of_schema_sentences": dropped}

    with open(DATA_PROC / "data_stats.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

    write_manifest("02_preprocess", DATA_PROC,
                   inputs=sorted(DATA_RAW.glob("*.jsonl")), extra=report)


if __name__ == "__main__":
    main()
