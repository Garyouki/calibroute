"""Human-readable rendering for audit reports."""

from __future__ import annotations


def _format(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown(report: dict[str, object]) -> str:
    overall = report["overall"]
    lines = [
        "# CalibRoute audit",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ("count", "accuracy", "risk", "mean_confidence", "ece", "auroc", "aurc"):
        lines.append(f"| {key} | {_format(overall.get(key))} |")
    lines.extend(
        [
            "",
            "## Selective operating points",
            "",
            "| Coverage | Risk | Accuracy | Confidence threshold |",
            "|---:|---:|---:|---:|",
        ]
    )
    for point in overall["selective"]:
        lines.append(
            f"| {_format(point['coverage'])} | {_format(point['risk'])} | "
            f"{_format(point['accuracy'])} | {_format(point['threshold'])} |"
        )
    lines.extend(
        [
            "",
            "> These measurements describe the supplied data. They are not a safety guarantee,",
            "> and thresholds must be revalidated after model, task, or distribution changes.",
            "",
        ]
    )
    return "\n".join(lines)
