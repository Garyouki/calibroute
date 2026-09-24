"""Shared utilities: traceability manifests, seeding, span/metric helpers.

Every pipeline script calls write_manifest() so each artifact records the
exact code version, config, inputs and environment that produced it.
"""
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = EXP_ROOT / "data" / "raw"
DATA_PROC = EXP_ROOT / "data" / "processed"
RUNS = EXP_ROOT / "runs"
RESULTS = EXP_ROOT / "results"
LOGS = EXP_ROOT / "logs"


def load_config():
    import yaml
    with open(EXP_ROOT / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=EXP_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def file_sha256(path, limit_mb=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(stage, out_dir, args=None, inputs=None, extra=None):
    """Record provenance for a pipeline stage next to its outputs."""
    import torch, transformers
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "command": " ".join(sys.argv),
        "args": args or {},
        "input_hashes": {str(p): file_sha256(p) for p in (inputs or []) if Path(p).exists()},
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "extra": extra or {},
    }
    path = out_dir / f"manifest_{stage}.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def set_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------- span helpers ----------------

def bio_to_spans(tags):
    """BIO tag list -> set-like list of (start, end_exclusive, type)."""
    spans, start, typ = [], None, None
    for i, t in enumerate(tags):
        if t.startswith("B-"):
            if start is not None:
                spans.append((start, i, typ))
            start, typ = i, t[2:]
        elif t.startswith("I-") and start is not None and t[2:] == typ:
            continue
        else:
            if start is not None:
                spans.append((start, i, typ))
            start, typ = None, None
            if t.startswith("I-"):  # ill-formed I- treated as B-
                start, typ = i, t[2:]
    if start is not None:
        spans.append((start, len(tags), typ))
    return spans


def norm_text(s):
    """Normalize entity surface form for position-less matching: lowercase,
    collapse whitespace, attach then strip boundary punctuation so that
    tokenization artifacts ('Inc .' vs 'Inc.') do not count as errors."""
    import re
    s = " ".join(s.strip().lower().split())
    s = re.sub(r"\s+([.,;:!?])", r"\1", s)
    return s.strip(".,;:!?'\"`")


def match_entities(gold, pred):
    """Position-less multiset matching on (normalized text, type).

    gold/pred: lists of (text, type). Returns (tp, fp, fn, matched_pred_idx).
    """
    from collections import Counter
    gold_c = Counter((norm_text(t), y) for t, y in gold)
    tp = 0
    matched = []
    for i, (t, y) in enumerate(pred):
        key = (norm_text(t), y)
        if gold_c.get(key, 0) > 0:
            gold_c[key] -= 1
            tp += 1
            matched.append(i)
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn, set(matched)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------- calibration / selective metrics ----------------

def ece(confs, correct, n_bins=15):
    """Expected calibration error over entity-level predictions."""
    import numpy as np
    confs, correct = np.asarray(confs, float), np.asarray(correct, float)
    if len(confs) == 0:
        return float("nan")
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - confs[m].mean())
    return float(e)


def auroc(confs, correct):
    import numpy as np
    from sklearn.metrics import roc_auc_score
    correct = np.asarray(correct)
    if len(set(correct.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(correct, confs))


def risk_coverage(confs, errors):
    """Sort by confidence desc; return coverage[], risk[] and AURC."""
    import numpy as np
    confs, errors = np.asarray(confs, float), np.asarray(errors, float)
    order = np.argsort(-confs)
    err_sorted = errors[order]
    n = len(errors)
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(err_sorted) / np.arange(1, n + 1)
    aurc = float(np.trapezoid(risk, cov))
    return cov, risk, aurc
