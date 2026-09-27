"""Reproduce the frozen synthetic shift-monitoring protocol with the released package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import calibroute
from calibroute.models import Action, PredictionRecord
from calibroute.policy import GatePolicy, route_batch
from calibroute.shift import calibrate_js_threshold, confidence_histogram, js_divergence

HERE = Path(__file__).resolve().parent


def stream(seed: int, *keys: object) -> random.Random:
    material = json.dumps([seed, *keys], separators=(",", ":")).encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


def wilson(successes: int, count: int) -> list[float]:
    """95% marginal Monte Carlo interval for independent simulation trials."""
    z = 1.959963984540054
    rate = successes / count
    scale = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / scale
    half = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / scale
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize_actions(rows, actions):
    accepted = [i for i, action in enumerate(actions) if action == Action.ACCEPT]
    return {
        "count": len(rows),
        "accepted": len(accepted),
        "accepted_errors": sum(not rows[i].correct for i in accepted),
        "review": sum(action == Action.HUMAN_REVIEW for action in actions),
        "abstain": sum(action == Action.ABSTAIN for action in actions),
        "total_errors": sum(not row.correct for row in rows),
    }


def make_rows(scores, replacements, label_uniforms, scenario):
    rows = []
    for index, (latent, replacement, label) in enumerate(zip(scores, replacements, label_uniforms)):
        observed = 0.05 if replacement < scenario["replacement_probability"] else latent
        signal = latent if scenario["label_rule"] == "latent_score" else observed
        probability = 0.55 + 0.45 * signal - scenario["accuracy_penalty"]
        if not 0 <= probability <= 1:
            raise ValueError("protocol generates invalid correctness probability")
        rows.append(PredictionRecord(str(index), observed, label < probability, "synthetic"))
    return rows


def study_cell(plan, profile, reference_size, batch_size, replications):
    """Pair scenarios; independently regenerate reference and batch on every trial."""
    gate = plan["gate"]
    weights = profile["weights"]
    population = [weight / sum(weights) for weight in weights]
    oracle_threshold = max(
        gate["effect_floor"],
        calibrate_js_threshold(
            population,
            batch_size,
            confidence_level=gate["confidence_level"],
            resamples=gate["resamples"],
            seed=gate["seed"],
        ),
    )
    scenarios = (
        plan["scenarios"]
        if (
            reference_size == plan["scenario_reference_size"]
            and batch_size == plan["scenario_batch_size"]
        )
        else plan["scenarios"][:1]
    )
    trials = []
    for replication in range(replications):
        keys = (profile["name"], reference_size, batch_size, replication)
        reference = stream(plan["seed"], *keys, "reference").choices(
            plan["bin_centers"], weights=weights, k=reference_size
        )
        scores = stream(plan["seed"], *keys, "batch").choices(
            plan["bin_centers"], weights=weights, k=batch_size
        )
        replacement_rng = stream(plan["seed"], *keys, "replacement")
        label_rng = stream(plan["seed"], *keys, "label")
        replacements = [replacement_rng.random() for _ in scores]
        labels = [label_rng.random() for _ in scores]
        reference_histogram = confidence_histogram(reference, bins=len(weights))
        policy = GatePolicy(
            accept_threshold=plan["routing"]["accept_threshold"],
            review_threshold=plan["routing"]["review_threshold"],
            max_validation_risk=0.0,
            validation_coverage=1.0,
            reference_histogram=reference_histogram,
            histogram_bins=len(weights),
            shift_confidence_level=gate["confidence_level"],
            shift_resamples=gate["resamples"],
            shift_effect_floor=gate["effect_floor"],
            shift_min_batch_size=gate["minimum_batch"],
            shift_seed=gate["seed"],
        )
        # Thresholds are predeclared, not fitted. Fit-only metadata above are unused.
        for scenario in scenarios:
            rows = make_rows(scores, replacements, labels, scenario)
            decisions, summary = route_batch(rows, policy)
            # Same reference and batch size imply exactly the same deterministic
            # calibration. Reuse its numeric value for paired scenarios only.
            policy = replace(policy, max_js_divergence=summary["max_js_divergence"])
            no_gate, _ = route_batch(rows, replace(policy, max_js_divergence=1.0))
            fixed = [
                Action.ACCEPT
                if r.confidence >= plan["routing"]["fixed_threshold"]
                else Action.HUMAN_REVIEW
                for r in rows
            ]
            metrics = {
                "fixed_090": summarize_actions(rows, fixed),
                "threshold_only": summarize_actions(rows, [d.action for d in no_gate]),
                "full_router": summarize_actions(rows, [d.action for d in decisions]),
            }
            observed = confidence_histogram((r.confidence for r in rows), bins=len(weights))
            trials.append(
                {
                    "profile": profile["name"],
                    "reference_size": reference_size,
                    "batch_size": batch_size,
                    "replication": replication,
                    "scenario": scenario["name"],
                    "alarm": summary["severe_shift"],
                    "shift_score": summary["shift_score"],
                    "threshold": summary["max_js_divergence"],
                    "oracle_alarm": js_divergence(population, observed) > oracle_threshold,
                    "oracle_threshold": oracle_threshold,
                    "reference_histogram": reference_histogram,
                    "observed_histogram": observed,
                    "metrics": metrics,
                }
            )
    return trials


def aggregate(trials):
    grouped = {}
    for trial in trials:
        key = (trial["profile"], trial["reference_size"], trial["batch_size"], trial["scenario"])
        grouped.setdefault(key, []).append(trial)
    results = []
    for (profile, reference_size, batch_size, scenario), rows in sorted(grouped.items()):
        alarms = sum(row["alarm"] for row in rows)
        oracle_alarms = sum(row["oracle_alarm"] for row in rows)
        metrics = {}
        for rule in rows[0]["metrics"]:
            totals = {
                field: sum(r["metrics"][rule][field] for r in rows)
                for field in rows[0]["metrics"][rule]
            }
            totals["coverage"] = totals["accepted"] / totals["count"]
            totals["review_rate"] = totals["review"] / totals["count"]
            totals["accepted_risk"] = (
                totals["accepted_errors"] / totals["accepted"] if totals["accepted"] else None
            )
            metrics[rule] = totals
        full, alone = metrics["full_router"], metrics["threshold_only"]
        extra_reviews = full["review"] - alone["review"]
        withheld = alone["accepted_errors"] - full["accepted_errors"]
        results.append(
            {
                "profile": profile,
                "reference_size": reference_size,
                "batch_size": batch_size,
                "scenario": scenario,
                "replications": len(rows),
                "alarms": alarms,
                "alarm_rate": alarms / len(rows),
                "alarm_rate_wilson_95": wilson(alarms, len(rows)),
                "oracle_alarms": oracle_alarms,
                "oracle_alarm_rate": oracle_alarms / len(rows),
                "oracle_rate_wilson_95": wilson(oracle_alarms, len(rows)),
                "mean_threshold": sum(r["threshold"] for r in rows) / len(rows),
                "floor_binding_trials": sum(r["threshold"] == 0.02 for r in rows),
                "metrics": metrics,
                "additional_reviews": extra_reviews,
                "accepted_errors_withheld": withheld,
                "extra_reviews_per_withheld_error": extra_reviews / withheld if withheld else None,
            }
        )
    return results


def render_report(report):
    lines = [
        "# Synthetic shift-monitoring study",
        "",
        f"Mode: {report['mode']}",
        "",
        f"Protocol SHA-256: `{report['protocol_sha256']}`",
        "",
        "All data are simulated. Intervals are marginal 95% Wilson Monte Carlo intervals.",
        "The population oracle knows the true score distribution; it is a diagnostic only.",
        "",
        "## Unchanged distributions",
        "",
        "| Profile | Reference N | Batch N | Alarms / trials | Alarm rate (95% interval) | Population oracle |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for cell in report["cells"]:
        if cell["scenario"] != "stable":
            continue
        low, high = cell["alarm_rate_wilson_95"]
        lines.append(
            f"| {cell['profile']} | {cell['reference_size']} | {cell['batch_size']} | "
            f"{cell['alarms']}/{cell['replications']} | {cell['alarm_rate']:.1%} "
            f"({low:.1%}–{high:.1%}) | {cell['oracle_alarm_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Paired scenarios: reference 150, batch 300",
            "",
            "| Profile | Scenario | Alarm rate | Fixed 0.90 coverage / risk | Threshold-only coverage / risk | Full coverage / risk | Full review rate | Additional reviews | Accepted errors withheld |",
            "|---|---|---:|---|---|---|---:|---:|---:|",
        ]
    )
    for cell in report["cells"]:
        if cell["reference_size"] != 150 or cell["batch_size"] != 300:
            continue
        pairs = []
        for rule in ("fixed_090", "threshold_only", "full_router"):
            metrics = cell["metrics"][rule]
            risk = (
                "undefined"
                if metrics["accepted_risk"] is None
                else f"{metrics['accepted_risk']:.2%}"
            )
            pairs.append(f"{metrics['coverage']:.2%} / {risk}")
        lines.append(
            f"| {cell['profile']} | {cell['scenario']} | {cell['alarm_rate']:.1%} | "
            + " | ".join(pairs)
            + f" | {cell['metrics']['full_router']['review_rate']:.2%} | "
            f"{cell['additional_reviews']} | {cell['accepted_errors_withheld']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Routing thresholds 0.90, 0.80, and review 0.60 were predeclared, not fitted.",
            "- Pooled routing counts are descriptive, not independent per-record confidence intervals.",
            "- Errors withheld means removed from automatic acceptance; it does not mean a reviewer corrected them.",
            "- The gate also sends previously abstained rows to review, so additional reviews include those rows.",
            "- Scenarios share random streams within a trial. Trials regenerate both reference and batch.",
            "- Stable alarm rates are marginal over new references, not a guarantee for each deployed reference.",
            "- Label-only changes preserve every score and must yield identical alarms to stable data.",
            "- Synthetic findings do not classify the historical FIN alarm as a false positive.",
            "- No defaults were optimized using these outcomes; no review time or production benefit was measured.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--output", type=Path, default=HERE.parent / "output/shift_study")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--replications", type=int, help="Override for a labeled smoke run only.")
    args = parser.parse_args(argv)
    plan_bytes = args.protocol.read_bytes()
    plan = json.loads(plan_bytes)
    replications = plan["replications"] if args.replications is None else args.replications
    if replications < 1 or args.workers < 1:
        parser.error("replications and workers must be positive")
    version = importlib.metadata.version("calibroute-ai")
    if version != plan["package_version"]:
        parser.error(f"protocol requires calibroute-ai=={plan['package_version']}, found {version}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty; preserve previous runs")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "protocol.json").write_bytes(plan_bytes)
    jobs = [
        (plan, profile, reference_size, batch_size, replications)
        for profile in plan["profiles"]
        for reference_size in plan["reference_sizes"]
        for batch_size in plan["batch_sizes"]
    ]
    trials = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(study_cell, *job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            trials.extend(rows)
            first = rows[0]
            print(
                f"Completed {index}/{len(jobs)}: {first['profile']} "
                f"reference={first['reference_size']} batch={first['batch_size']}",
                flush=True,
            )
    trials.sort(
        key=lambda r: (
            r["profile"],
            r["reference_size"],
            r["batch_size"],
            r["replication"],
            r["scenario"],
        )
    )
    trial_path = args.output / "trials.jsonl"
    with trial_path.open("w", encoding="utf-8", newline="\n") as handle:
        for trial in trials:
            handle.write(json.dumps(trial, sort_keys=True) + "\n")
    report = {
        "schema_version": "1.0",
        "mode": "full_protocol" if args.replications is None else "smoke",
        "protocol_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trials_sha256": hashlib.sha256(trial_path.read_bytes()).hexdigest(),
        "package_version": version,
        "python": platform.python_version(),
        "package_location": calibroute.__file__,
        "executable": sys.executable,
        "trial_scenario_count": len(trials),
        "replications_per_cell": replications,
        "cells": aggregate(trials),
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(render_report(report), encoding="utf-8")
    print(f"Wrote {len(trials)} trial-scenarios to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
