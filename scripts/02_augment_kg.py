import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-kg", type=Path, required=True, help="Folder containing kg.csv and node.csv")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output-kg", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    base_kg, base_nodes = args.base_kg / "kg.csv", args.base_kg / "node.csv"
    if not base_kg.exists() or not base_nodes.exists():
        raise FileNotFoundError("--base-kg must contain kg.csv and node.csv")
    if args.output_kg.exists():
        raise FileExistsError(f"Refusing to overwrite existing graph: {args.output_kg}")

    # Copy the original graph.
    args.output_kg.mkdir(parents=True)
    out_kg, out_nodes = args.output_kg / "kg.csv", args.output_kg / "node.csv"
    shutil.copy2(base_kg, out_kg)
    shutil.copy2(base_nodes, out_nodes)

    # Append DEG-derived edges and disease-state/cell-type nodes.
    added_edges = 0
    for filename in ("gene_to_state_cell_edges.csv", "cra_to_state_cell_edges.csv"):
        edges = pd.read_csv(args.evidence_dir / filename)
        edges.to_csv(out_kg, mode="a", index=False, header=False)
        added_edges += len(edges)
    added_nodes = pd.read_csv(args.evidence_dir / "added_state_cell_nodes.tsv", sep="\t")
    added_nodes.to_csv(out_nodes, sep="\t", mode="a", index=False, header=False)

    manifest = {
        "base_kg": str(args.base_kg), "added_nodes": int(len(added_nodes)),
        "added_edges": int(added_edges), "purpose": "CRA DEG-derived state-cell graph augmentation",
        "note": "The base graph is copied and remains unchanged.",
    }
    (args.output_kg / "augmentation_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
