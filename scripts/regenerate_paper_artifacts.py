#!/usr/bin/env python3
"""Regenerate paper-facing tables and main figures from public artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


METHODS = (
    "StreamingLLMPress",
    "SnapKVPress",
    "TOVAPress",
    "KeyDiffPress",
    "ThinKPress",
    "ChunkKVPress_Knorm",
    "AdaKVPress",
    "QuantizedCache",
)
METHOD_LABEL = {
    "StreamingLLMPress": "StreamingLLM",
    "SnapKVPress": "SnapKV",
    "TOVAPress": "TOVA",
    "KeyDiffPress": "KeyDiff",
    "ThinKPress": "ThinK",
    "ChunkKVPress_Knorm": "ChunkKV",
    "AdaKVPress": "AdaKV",
    "QuantizedCache": "QuantizedCache",
}
DISPLAY_LABELS = (
    "StreamLLM",
    "SnapKV",
    "TOVA",
    "KeyDiff",
    "ThinK",
    "ChunkKV",
    "AdaKV",
    "Quant.",
)
ECOV_LABELS = (
    "StreamLLM",
    "SnapKV",
    "TOVA",
    "KeyDiff",
    "ThinK",
    "ChunkKV",
    "AdaKV",
    "Quant.",
)
DATASETS = ("ruler8k", "ruler16k", "qasper", "hotpotqa")
DATASET_LABEL = {
    "ruler8k": "RULER-8K",
    "ruler16k": "RULER-16K",
    "qasper": "Qasper",
    "hotpotqa": "HotpotQA",
}
BUDGETS = (0.75, 0.50, 0.25)
QA_SIGNATURES = (
    ("low_mapped_coverage", "Low mapped coverage"),
    ("partial_mapped_coverage", "Partial mapped coverage"),
    ("high_mapped_coverage_likelihood_drift", "Mapped-position drift"),
    ("structural_position_likelihood_drift", "Structural-position drift"),
    ("decode_scorer_candidate", "Decoding/scoring"),
    ("ambiguous", "Ambiguous"),
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if path.name.endswith(".jsonl.gz"):
        handle = gzip.open(path, mode="rt", encoding="utf-8")
    else:
        handle = path.open(encoding="utf-8")
    with handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def percent(value: float | None) -> float | None:
    return None if value is None else 100.0 * value


def normalize_budget(value: Any) -> float:
    return round(float(value), 6)


def load_public_data(root: Path):
    population_dir = root / "data/processed/full_population"
    fullcache = {
        row["fullcache_key"]: row
        for row in iter_jsonl(population_dir / "fullcache.jsonl.gz")
    }
    compressed: list[dict[str, Any]] = []
    for dataset in DATASETS:
        compressed.extend(
            iter_jsonl(population_dir / "compressed_runs" / f"{dataset}.jsonl.gz")
        )
    selected = list(
        iter_jsonl(root / "data/processed/selected_failures/all_selected_failures.jsonl")
    )
    return fullcache, compressed, selected


def build_tables(root: Path, output_dir: Path):
    fullcache, compressed, selected = load_public_data(root)
    supported = [row for row in compressed if row["support_status"] == "supported"]

    full_evaluation_rows: list[dict[str, Any]] = []
    cell_values: dict[tuple[str, str, float], dict[str, Any]] = {}
    for dataset in DATASETS:
        full_rows = [row for row in fullcache.values() if row["dataset"] == dataset]
        for method in METHODS:
            for budget in BUDGETS:
                rows = [
                    row
                    for row in compressed
                    if row["dataset"] == dataset
                    and row["method_name"] == method
                    and normalize_budget(row["retained_budget"]) == budget
                ]
                supported_rows = [row for row in rows if row["support_status"] == "supported"]
                eligible = [
                    row for row in supported_rows if fullcache[row["fullcache_key"]]["full_correct"]
                ]
                c_to_w = [row for row in supported_rows if row["outcome_transition"] == "C->W"]
                record = {
                    "dataset": dataset,
                    "dataset_display": DATASET_LABEL[dataset],
                    "method_name": method,
                    "method_display": METHOD_LABEL[method],
                    "retained_budget": budget,
                    "source_count": len(full_rows),
                    "support_status": "unsupported" if not supported_rows else "supported",
                    "fullcache_task_score": percent(mean(row["full_score"] for row in full_rows)),
                    "fullcache_binary_accuracy": percent(
                        mean(row["full_correct"] for row in full_rows)
                    ),
                    "fullcache_correct_sources": sum(bool(row["full_correct"]) for row in full_rows),
                    "compressed_task_score": percent(
                        mean(row["compressed_score"] for row in supported_rows)
                    ),
                    "compressed_binary_accuracy": percent(
                        mean(row["compressed_correct"] for row in supported_rows)
                    ),
                    "C_to_W_rows": len(c_to_w),
                    "C_to_W_rate_given_fullcache_correct": percent(
                        len(c_to_w) / len(eligible) if eligible else None
                    ),
                }
                full_evaluation_rows.append(record)
                cell_values[(dataset, method, budget)] = record
    write_csv(output_dir / "full_evaluation_results.csv", full_evaluation_rows)

    pooled_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for budget in BUDGETS:
            rows = [
                row
                for row in supported
                if row["method_name"] == method
                and normalize_budget(row["retained_budget"]) == budget
            ]
            eligible = [row for row in rows if fullcache[row["fullcache_key"]]["full_correct"]]
            c_to_w = [row for row in rows if row["outcome_transition"] == "C->W"]
            pooled_rows.append(
                {
                    "method_name": method,
                    "method_display": METHOD_LABEL[method],
                    "retained_budget": budget,
                    "support_status": "unsupported" if not rows else "supported",
                    "compressed_task_score": percent(
                        mean(row["compressed_score"] for row in rows)
                    ),
                    "fullcache_correct_pairs": len(eligible),
                    "C_to_W_rows": len(c_to_w),
                    "C_to_W_rate_given_fullcache_correct": percent(
                        len(c_to_w) / len(eligible) if eligible else None
                    ),
                }
            )
    write_csv(output_dir / "pooled_population_outcomes.csv", pooled_rows)

    failure_view_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for budget in BUDGETS:
            rates = [
                cell_values[(dataset, method, budget)][
                    "C_to_W_rate_given_fullcache_correct"
                ]
                for method in METHODS
                if cell_values[(dataset, method, budget)]["support_status"] == "supported"
            ]
            failure_sets: dict[str, set[str]] = {}
            for method in ("SnapKVPress", "TOVAPress"):
                failure_sets[method] = {
                    str(row["sample_id"])
                    for row in supported
                    if row["dataset"] == dataset
                    and row["method_name"] == method
                    and normalize_budget(row["retained_budget"]) == budget
                    and row["outcome_transition"] == "C->W"
                }
            union = failure_sets["SnapKVPress"] | failure_sets["TOVAPress"]
            failure_view_rows.append(
                {
                    "dataset": dataset,
                    "dataset_display": DATASET_LABEL[dataset],
                    "retained_budget": budget,
                    "mean_C_to_W_rate": mean(rates),
                    "SnapKV_TOVA_failure_Jaccard": len(
                        failure_sets["SnapKVPress"] & failure_sets["TOVAPress"]
                    )
                    / len(union)
                    if union
                    else None,
                }
            )
    write_csv(output_dir / "failure_views.csv", failure_view_rows)

    diagnostic_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for budget in BUDGETS:
            rows = [
                row
                for row in selected
                if row["method_name"] == method
                and normalize_budget(row["retained_budget"]) == budget
            ]
            diagnostic_rows.append(
                {
                    "method_name": method,
                    "method_display": METHOD_LABEL[method],
                    "retained_budget": budget,
                    "failure_rows": len(rows),
                    "ECov_slot_mean": mean(row.get("ECov_slot") for row in rows),
                    "delta_NLL_mean": mean(row.get("delta_NLL") for row in rows),
                    "GPR_mean": mean(row.get("GPR") for row in rows),
                    "EAR_mean": mean(
                        row.get("EAR") for row in rows if row.get("attention_available")
                    ),
                }
            )
    write_csv(output_dir / "diagnostic_profiles.csv", diagnostic_rows)

    qa_rows: list[dict[str, Any]] = []
    for dataset in ("qasper", "hotpotqa"):
        counts = Counter(
            row["failure_signature"] for row in selected if row["dataset"] == dataset
        )
        for signature, label in QA_SIGNATURES:
            qa_rows.append(
                {
                    "dataset": dataset,
                    "dataset_display": DATASET_LABEL[dataset],
                    "failure_signature": signature,
                    "signature_display": label,
                    "rows": counts[signature],
                    "share_percent": 100.0
                    * counts[signature]
                    / sum(counts.values()),
                }
            )
    write_csv(output_dir / "qa_signature_counts.csv", qa_rows)

    summary = json.loads(
        (root / "data/processed/full_population/summary.json").read_text(encoding="utf-8")
    )
    (output_dir / "run_accounting.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return pooled_rows, failure_view_rows, diagnostic_rows, qa_rows, selected


def matrix(rows, row_key, value_key):
    result = []
    for name in row_key:
        result.append(
            [
                next(
                    (
                        row[value_key]
                        for row in rows
                        if row.get("method_name", row.get("dataset")) == name
                        and normalize_budget(row["retained_budget"]) == budget
                    ),
                    None,
                )
                for budget in BUDGETS
            ]
        )
    return result


def plot_artifacts(output_dir, assets_dir, pooled, failure_views, diagnostics, qa_rows, selected):
    import matplotlib as mpl
    import numpy as np

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.1,
            "axes.labelsize": 8.1,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    def as_array(values):
        return np.asarray(
            [[np.nan if value is None else value for value in row] for row in values],
            dtype=float,
        )

    def annotate(ax, image, values, fmt):
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                if not np.isfinite(value):
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=7.0, color="0.35")
                    continue
                rgba = image.cmap(image.norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(
                    j,
                    i,
                    fmt.format(value),
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color="black" if luminance > 0.52 else "white",
                )

    def heatmap_style(ax, nrows):
        ax.set_xticks(range(3), ["75", "50", "25"])
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.45, alpha=0.72)
        ax.tick_params(which="minor", bottom=False, left=False)

    def save(fig, stem):
        pdf = output_dir / f"{stem}.pdf"
        png = output_dir / f"{stem}.png"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=400, bbox_inches="tight")
        plt.close(fig)
        if assets_dir is not None:
            assets_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(png, assets_dir / f"{stem}.png")

    score = as_array(matrix(pooled, METHODS, "compressed_task_score"))
    ctw = as_array(matrix(pooled, METHODS, "C_to_W_rate_given_fullcache_correct"))
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.04), constrained_layout=True, sharey=True)
    score_img = axes[0].imshow(score, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    ctw_img = axes[1].imshow(ctw, vmin=0, vmax=100, cmap="YlOrRd", aspect="auto")
    axes[0].set_yticks(range(len(METHODS)), DISPLAY_LABELS)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    annotate(axes[0], score_img, score, "{:.1f}")
    annotate(axes[1], ctw_img, ctw, "{:.1f}")
    for label, title, ax in zip("AB", ("Mean task score $\\uparrow$", "C$\\rightarrow$W rate (%) $\\downarrow$"), axes):
        heatmap_style(ax, len(METHODS))
        ax.set_title(rf"$\bf{{{label}}}$  {title}", loc="left", pad=2)
    fig.supxlabel("Compression setting (%)", fontsize=7.7)
    save(fig, "population-outcomes")

    failure_rate = as_array(
        [
            [
                next(
                    row["mean_C_to_W_rate"]
                    for row in failure_views
                    if row["dataset"] == dataset and row["retained_budget"] == budget
                )
                for budget in BUDGETS
            ]
            for dataset in DATASETS
        ]
    )
    jaccard = as_array(
        [
            [
                next(
                    row["SnapKV_TOVA_failure_Jaccard"]
                    for row in failure_views
                    if row["dataset"] == dataset and row["retained_budget"] == budget
                )
                for budget in BUDGETS
            ]
            for dataset in DATASETS
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.62), constrained_layout=True, sharey=True)
    rate_img = axes[0].imshow(failure_rate, vmin=0, vmax=50, cmap="YlOrRd", aspect="auto")
    jac_img = axes[1].imshow(jaccard, vmin=0, vmax=0.75, cmap="YlGnBu", aspect="auto")
    axes[0].set_yticks(range(len(DATASETS)), [DATASET_LABEL[d] for d in DATASETS])
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    annotate(axes[0], rate_img, failure_rate, "{:.0f}")
    annotate(axes[1], jac_img, jaccard, "{:.2f}")
    for label, title, ax in zip("AB", ("C$\\rightarrow$W rate (%)", "Failure-set Jaccard"), axes):
        heatmap_style(ax, len(DATASETS))
        ax.set_title(rf"$\bf{{{label}}}$  {title}", loc="left", pad=2)
    fig.supxlabel("Compression setting (%)", fontsize=7.7)
    save(fig, "failure-views")

    ecov = as_array(matrix(diagnostics, METHODS, "ECov_slot_mean"))
    dnll = as_array(matrix(diagnostics, METHODS, "delta_NLL_mean"))
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.00), constrained_layout=True, sharey=True)
    ecov_img = axes[0].imshow(ecov, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    dnll_img = axes[1].imshow(dnll, vmin=0, vmax=8, cmap="YlOrRd", aspect="auto")
    axes[0].set_yticks(range(len(METHODS)), ECOV_LABELS)
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    annotate(axes[0], ecov_img, ecov, "{:.2f}")
    annotate(axes[1], dnll_img, dnll, "{:.1f}")
    for label, title, ax in zip("AB", ("Mapped slot ECov", "Gold-answer $\\Delta$NLL"), axes):
        heatmap_style(ax, len(METHODS))
        ax.set_title(rf"$\bf{{{label}}}$  {title}", loc="left", pad=2)
    fig.supxlabel("Compression setting (%)", fontsize=7.7)
    save(fig, "diagnostic-profiles")

    categories = [label for _, label in QA_SIGNATURES]
    fig, ax = plt.subplots(figsize=(4.65, 2.15), constrained_layout=True)
    ypos = np.arange(len(categories), dtype=float)
    styles = {"qasper": ("#CC79A7", "D", 5.5), "hotpotqa": ("#D55E00", "^", -7.0)}
    values = {}
    for dataset in ("qasper", "hotpotqa"):
        rows = [row for row in qa_rows if row["dataset"] == dataset]
        values[dataset] = np.asarray([row["share_percent"] for row in rows])
    for index in range(len(categories)):
        ax.plot([values["qasper"][index], values["hotpotqa"][index]], [index, index], color="0.72", linewidth=0.8)
    for dataset in ("qasper", "hotpotqa"):
        color, marker, offset = styles[dataset]
        counts = [row["rows"] for row in qa_rows if row["dataset"] == dataset]
        total = sum(counts)
        ax.scatter(values[dataset], ypos, s=26, color=color, marker=marker, edgecolors="white", linewidths=0.45, label=f"{DATASET_LABEL[dataset]} (n={total})", zorder=3)
        for xvalue, yvalue, count in zip(values[dataset], ypos, counts):
            ax.annotate(str(count), (xvalue, yvalue), xytext=(0, offset), textcoords="offset points", ha="center", va="center", fontsize=6.6)
    ax.set_yticks(ypos, categories)
    ax.invert_yaxis()
    ax.set_xlim(-2, 56)
    ax.set_xlabel(r"Share of C$\rightarrow$W rows (%)")
    ax.grid(axis="x", color="0.88", linewidth=0.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Diagnostic-category composition", loc="left", pad=27)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=2,
        frameon=False,
        borderaxespad=0,
        handletextpad=0.45,
        columnspacing=1.2,
    )

    save(fig, "qa-transfer")

    qasper_case = next(
        row
        for row in selected
        if row["dataset"] == "qasper"
        and row["sample_id"] == "qasper_000196"
        and row["method_name"] == "ThinKPress"
        and row["retained_budget"] == 0.5
    )
    hotpot_case = next(
        row
        for row in selected
        if row["dataset"] == "hotpotqa"
        and row["sample_id"] == "hotpotqa_000076"
        and row["method_name"] == "ChunkKVPress_Knorm"
        and row["retained_budget"] == 0.5
    )
    fig, case_ax = plt.subplots(figsize=(3.35, 1.62), constrained_layout=True)
    case_ax.set_xlim(0, 1)
    case_ax.set_ylim(0, 1)
    case_ax.axis("off")
    case_ax.plot([0.018, 0.018], [0.54, 0.96], color="#CC79A7", linewidth=2.4)
    case_ax.plot([0.018, 0.018], [0.04, 0.46], color="#D55E00", linewidth=2.4)
    case_ax.plot([0.018, 0.99], [0.50, 0.50], color="0.84", linewidth=0.55)
    case_ax.text(
        0.045,
        0.90,
        "Qasper | ThinK, 50% | structural-position drift",
        fontsize=7.2,
        fontweight="bold",
        va="center",
    )
    case_ax.text(
        0.045,
        0.72,
        r'FullCache: "Groningen Meaning Bank"  $\rightarrow$  "The The $\ldots$"',
        fontsize=6.9,
        va="center",
    )
    case_ax.text(
        0.045,
        0.58,
        rf"ECov N/A; $\Delta$NLL={qasper_case['delta_NLL']:.3f}. Positions remain addressable; fidelity is unknown.",
        fontsize=6.5,
        va="center",
        color="0.23",
    )
    case_ax.text(
        0.045,
        0.40,
        "HotpotQA | ChunkKV, 50% | partial projected coverage",
        fontsize=7.2,
        fontweight="bold",
        va="center",
    )
    case_ax.text(
        0.045,
        0.22,
        rf"FullCache: {hotpot_case['extracted_answer_full']}  $\rightarrow$  compressed: {hotpot_case['extracted_answer_compressed']}",
        fontsize=6.9,
        va="center",
    )
    case_ax.text(
        0.045,
        0.08,
        rf"ECov={hotpot_case['ECov_slot']:.3f}; $\Delta$NLL={hotpot_case['delta_NLL']:.3f}. Only part of the support chain remains.",
        fontsize=6.5,
        va="center",
        color="0.23",
    )
    save(fig, "qa-cases")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--assets-dir", type=Path)
    parser.add_argument("--tables-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "paper_artifacts/generated").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = build_tables(root, output_dir)
    if not args.tables_only:
        plot_artifacts(output_dir, args.assets_dir, *results)
    print(json.dumps({"ok": True, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
