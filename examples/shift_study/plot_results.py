"""Optional scientific figures; requires matplotlib (not a CalibRoute dependency)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

NAMES = {
    "uniform": "Uniform",
    "high_confidence": "95% in highest bin",
    "three_modes": "Three modes",
}
COLORS = ["#2459a6", "#cc6c24", "#29816e"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3), sharey=True)
    for axis, (profile, label) in zip(axes, NAMES.items()):
        for batch, color in zip((50, 300), COLORS):
            rows = sorted(
                [
                    c
                    for c in report["cells"]
                    if c["profile"] == profile
                    and c["scenario"] == "stable"
                    and c["batch_size"] == batch
                ],
                key=lambda c: c["reference_size"],
            )
            rates = [c["alarm_rate"] for c in rows]
            errors = [
                [max(0.0, c["alarm_rate"] - c["alarm_rate_wilson_95"][0]) for c in rows],
                [max(0.0, c["alarm_rate_wilson_95"][1] - c["alarm_rate"]) for c in rows],
            ]
            axis.errorbar(
                [0, 1, 2],
                rates,
                yerr=errors,
                marker="o",
                capsize=4,
                color=color,
                label=f"Batch {batch}",
            )
        axis.set(title=label, xlabel="Reference sample size", ylim=(0, 0.85))
        axis.set_xticks([0, 1, 2], ["50", "150", "500"])
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Alarm rate under unchanged distribution")
    axes[-1].legend(frameon=False)
    fig.suptitle("Finite reference samples can produce frequent alarms", fontsize=15)
    fig.text(
        0.5,
        0.02,
        f"Synthetic data | {report['replications_per_cell']} trials/cell | "
        "95% Wilson Monte Carlo intervals | CalibRoute 0.4.0 defaults",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"false-alarms.{suffix}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    scenarios = ["stable", "score_shift_05", "score_shift_15", "score_shift_30", "score_shift_60"]
    for (profile, label), color in zip(NAMES.items(), COLORS):
        cells = {
            c["scenario"]: c
            for c in report["cells"]
            if c["profile"] == profile and c["reference_size"] == 150 and c["batch_size"] == 300
        }
        axes[0].plot(
            [0, 5, 15, 30, 60],
            [cells[s]["alarm_rate"] for s in scenarios],
            color=color,
            marker="o",
            label=label,
        )
    axes[0].set(
        title="Controlled score changes",
        xlabel="Scores replaced with 0.05 (%)",
        ylabel="Alarm rate",
        ylim=(-0.03, 1.06),
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(frameon=False, fontsize=9)
    cells = {
        c["scenario"]: c
        for c in report["cells"]
        if c["profile"] == "high_confidence"
        and c["reference_size"] == 150
        and c["batch_size"] == 300
    }
    positions = [0, 1]
    axes[1].bar(
        [p - 0.18 for p in positions],
        [cells[s]["alarm_rate"] for s in ("stable", "label_only")],
        width=0.35,
        label="Alarm rate",
        color=COLORS[0],
    )
    axes[1].bar(
        [p + 0.18 for p in positions],
        [cells[s]["metrics"]["full_router"]["accepted_risk"] for s in ("stable", "label_only")],
        width=0.35,
        label="Full-router accepted error rate",
        color=COLORS[1],
    )
    axes[1].set_xticks(positions, ["Stable labels", "Correctness probability\nreduced by 0.30"])
    axes[1].set(title="Identical scores conceal accuracy loss", ylim=(0, 0.42))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, fontsize=9, loc="upper left")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Detection depends on score changes", fontsize=15)
    fig.text(
        0.5,
        0.02,
        "Synthetic data | Reference 150, batch 300 | Right panel: 95% in highest bin | "
        "Pooled accepted error rates",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"detection-and-blind-spot.{suffix}", dpi=180)
    plt.close(fig)
    print(f"Wrote figures to {args.output}; matplotlib {matplotlib.__version__}")


if __name__ == "__main__":
    main()
