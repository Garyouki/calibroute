"""Stage 05: LoRA fine-tune Qwen2.5-0.5B-Instruct for generative NER on FIN.

Output format the model is trained to emit:
  {"entities": [{"text": "...", "type": "PER|ORG|LOC"}, ...]}

Usage: python 05_train_generative.py --seed 42
Writes runs/generative_seed{SEED}/ with LoRA adapter + manifest.
"""
import argparse
import json
import time
import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from common import DATA_PROC, RUNS, load_config, set_seed, get_device, write_manifest, bio_to_spans

SYSTEM = ("You are a financial named-entity recognition system. Extract all "
          "person (PER), organization (ORG) and location (LOC) entities from "
          "the sentence. Respond with only a JSON object of the form "
          '{"entities": [{"text": "...", "type": "..."}]}. '
          'If there are no entities, respond {"entities": []}.')


def build_example(tokens, tags):
    text = " ".join(tokens)
    ents = [{"text": " ".join(tokens[s:e]), "type": t}
            for s, e, t in bio_to_spans(tags)]
    target = json.dumps({"entities": ents})
    return text, target


def make_messages(text):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Sentence: {text}"}]


def encode(rows, tok, max_length):
    feats = []
    for r in rows:
        text, target = build_example(r["tokens"], r["tags"])
        prompt_text = tok.apply_chat_template(make_messages(text), tokenize=False,
                                              add_generation_prompt=True)
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        target_ids = tok(target + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (prompt_ids + target_ids)[:max_length]
        labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]
        feats.append({"input_ids": ids, "labels": labels})
    return feats


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        d = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * d)
        labels.append(b["labels"] + [-100] * d)
        attn.append([1] * (n - d) + [0] * d)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config()["generative"]
    set_seed(args.seed)
    device = get_device()

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"],
                                                 dtype=torch.float16).to(device)
    lora = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                      lora_dropout=cfg["lora_dropout"],
                      target_modules=cfg["target_modules"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora)
    # fp16 base for memory/speed on MPS; LoRA params in fp32 for stable updates
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    with open(DATA_PROC / "fin_train.jsonl") as f:
        rows = [json.loads(l) for l in f]
    feats = encode(rows, tok, cfg["max_length"])
    g = torch.Generator().manual_seed(args.seed)
    dl = DataLoader(feats, batch_size=cfg["batch_size"], shuffle=True, generator=g,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    opt_steps = (len(dl) // cfg["grad_accum"] + 1) * cfg["epochs"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["lr"])
    sched = get_linear_schedule_with_warmup(opt, int(opt_steps * 0.05), opt_steps)

    out_dir = RUNS / f"generative_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log, t0 = [], time.time()
    model.train()
    step = 0
    for ep in range(cfg["epochs"]):
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="mps", dtype=torch.float16):
                out = model(**batch)
            loss = out.loss.float() / cfg["grad_accum"]
            loss.backward()
            if (i + 1) % cfg["grad_accum"] == 0:
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    log.append({"epoch": ep, "opt_step": step,
                                "loss": float(loss) * cfg["grad_accum"]})
                    print(f"ep{ep} opt_step{step} loss {float(loss)*cfg['grad_accum']:.4f}",
                          flush=True)
        opt.step(); sched.step(); opt.zero_grad()

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    write_manifest("05_train_generative", out_dir, args=vars(args) | cfg,
                   inputs=[DATA_PROC / "fin_train.jsonl"],
                   extra={"train_minutes": (time.time() - t0) / 60,
                          "final_loss": log[-1]["loss"] if log else None})
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
