import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Define the metrics and treatment relations.
METRICS = ("AUROC", "AUPRC", "Recall@10", "MRR@10", "AP@10")
ALIASES = {
    "relation": ("relation", "evaluation"),
    "disease_id": (
        "disease_id",
        "disease_idx",
        "disease_index",
        "disease_node_index",
        "disease",
        "disease_name",
    ),
    "AUROC": ("AUROC", "auroc"),
    "AUPRC": ("AUPRC", "auprc"),
    "Recall@10": ("Recall@10", "recall@10"),
    "MRR@10": ("MRR@10", "mrr@10"),
    "AP@10": ("AP@10", "ap@10"),
}
RELATIONS = ("contraindication", "indication", "off-label use")
RELATION_LABELS = ("Contraindication", "Indication", "Off-label use")


# Define the two result files and output location.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Original versus CRA-augmented TxGNN")
    return parser.parse_args()

#Normalize result-table column names.
def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for target, names in ALIASES.items():
        found = next((name for name in names if name in frame.columns), None)
        if found is None:
            raise ValueError(f"Missing {target}; expected one of {names}")
        rename[found] = target
    normalized = frame.rename(columns=rename).copy()
    normalized["relation"] = normalized["relation"].astype(str).str.removeprefix("rev_")
    normalized["disease_id"] = normalized["disease_id"].astype(str)
    return normalized


def match_runs(original: pd.DataFrame, augmented: pd.DataFrame) -> pd.DataFrame:
    # Pairing prevents a graph from looking better because it used easier diseases.
    keys = ["relation", "disease_id"]
    for label, frame in (("original", original), ("augmented", augmented)):
        duplicates = frame.duplicated(keys, keep=False)
        if duplicates.any():
            examples = frame.loc[duplicates, keys].drop_duplicates().head(5).to_dict("records")
            raise ValueError(f"{label} results contain duplicate relation/disease rows: {examples}")

    paired = original.merge(
        augmented,
        on=keys,
        how="outer",
        suffixes=("_original", "_augmented"),
        indicator=True,
        validate="one_to_one",
    )
    unmatched = paired.loc[paired["_merge"].ne("both"), keys + ["_merge"]]
    if not unmatched.empty:
        examples = unmatched.head(10).to_dict("records")
        raise ValueError(
            "The runs are not matched on the same held-out diseases. "
            f"Examples of unmatched rows: {examples}"
        )
    return paired.drop(columns="_merge")


def summarize_pairs(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for relation in RELATIONS:
        subset = paired.loc[paired["relation"].eq(relation)]
        if subset.empty:
            continue
        for metric in METRICS:
            original_mean = subset[f"{metric}_original"].mean()
            augmented_mean = subset[f"{metric}_augmented"].mean()
            rows.append(
                {
                    "relation": relation,
                    "metric": metric,
                    "n_diseases": int(len(subset)),
                    "original_mean": original_mean,
                    "augmented_mean": augmented_mean,
                    "mean_delta_augmented_minus_original": augmented_mean - original_mean,
                }
            )
    return pd.DataFrame(rows)

#Run the comparison and create the summary figure.
def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original = normalize_columns(pd.read_csv(args.original))
    augmented = normalize_columns(pd.read_csv(args.augmented))
    original = original.loc[original["relation"].isin(RELATIONS)].copy()
    augmented = augmented.loc[augmented["relation"].isin(RELATIONS)].copy()

    paired = match_runs(original, augmented)
    paired.to_csv(args.output_dir / "paired_disease_metrics.csv", index=False)
    summarize_pairs(paired).to_csv(args.output_dir / "paired_metric_summary.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    x = np.arange(len(RELATIONS))
    width = 0.35

    for axis, metric in zip(axes, METRICS):
        left = paired.groupby("relation")[f"{metric}_original"].mean().reindex(RELATIONS)
        right = paired.groupby("relation")[f"{metric}_augmented"].mean().reindex(RELATIONS)
        a = axis.bar(x - width / 2, left, width, label="Original TxGNN")
        b = axis.bar(x + width / 2, right, width, label="CRA-augmented TxGNN")
        axis.bar_label(a, fmt="%.3f", padding=3, fontsize=8)
        axis.bar_label(b, fmt="%.3f", padding=3, fontsize=8)
        axis.set_title(metric, weight="bold")
        axis.set_xticks(x, RELATION_LABELS)
        axis.set_ylabel("Mean disease-level score")
        axis.grid(axis="y", alpha=0.25)

    axes[-1].axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower right", bbox_to_anchor=(0.98, 0.05))
    fig.suptitle(args.title, fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    destination = args.output_dir / "original_vs_cra_augmented_metrics.png"
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
