import argparse
import json
from pathlib import Path

import pandas as pd

# Handle the column names in the DEG files and define the required table fields.
GENE_CANDIDATES = ("gene_symbol", "gene", "symbol", "Gene", "index", "Unnamed: 0")
PADJ_CANDIDATES = ("p_val_adj", "padj", "adjusted_p_value", "p_adj")
LFC_CANDIDATES = ("avg_log2FC", "log2FoldChange", "log2fc", "log2FC")
MANIFEST_COLUMNS = {"de_file", "comparison_id", "state_cell_node_name", "disease_state", "cell_type"}
NODE_COLUMNS = {"node_index", "node_id", "node_type", "node_name", "node_source"}


def pick_column(frame: pd.DataFrame, candidates: tuple[str, ...], label: str) -> str:
    found = next((name for name in candidates if name in frame.columns), None)
    if found is None:
        raise ValueError(f"Could not find {label}. Expected one of {candidates}; found {list(frame.columns)}")
    return found


def validate_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")

# Filter and rank DEG table.
def filter_deg_result(
    result: pd.DataFrame,
    *,
    fdr: float,
    min_abs_log2fc: float,
    direction: str,
    top_genes: int,
) -> pd.DataFrame:
    
    gene_col = pick_column(result, GENE_CANDIDATES, "gene-symbol column")
    padj_col = pick_column(result, PADJ_CANDIDATES, "adjusted-P column")
    lfc_col = pick_column(result, LFC_CANDIDATES, "log2-fold-change column")

    normalized = result[[gene_col, padj_col, lfc_col]].rename(
        columns={gene_col: "gene_symbol", padj_col: "p_val_adj", lfc_col: "avg_log2FC"}
    )
    normalized["gene_symbol"] = normalized["gene_symbol"].astype("string").str.strip()
    normalized["p_val_adj"] = pd.to_numeric(normalized["p_val_adj"], errors="coerce")
    normalized["avg_log2FC"] = pd.to_numeric(normalized["avg_log2FC"], errors="coerce")
    normalized = normalized.dropna(subset=["gene_symbol", "p_val_adj", "avg_log2FC"]).copy()
    normalized = normalized.loc[normalized["gene_symbol"].ne("")].copy()
    normalized["abs_log2FC"] = normalized["avg_log2FC"].abs()

    keep = (normalized["p_val_adj"] < fdr) & (normalized["abs_log2FC"] >= min_abs_log2fc)
    if direction == "disease_enriched":
        keep &= normalized["avg_log2FC"] <= -min_abs_log2fc
    elif direction != "both":
        raise ValueError(f"Unsupported direction: {direction}")

    filtered = normalized.loc[keep].copy()

    return filtered.sort_values(
        ["p_val_adj", "abs_log2FC", "gene_symbol"],
        ascending=[True, False, True],
        kind="mergesort",
    ).head(top_genes)


#Build a TxGNN gene lookup table
# The DEG files contain gene symbols, whereas the KG uses node indices/IDs.
# This lookup connects those two representations.
def build_gene_map(nodes: pd.DataFrame) -> pd.DataFrame:
    validate_columns(nodes, NODE_COLUMNS, "node.csv")
    gene_nodes = nodes.loc[nodes["node_type"].eq("gene/protein")].copy()
    gene_nodes["node_name"] = gene_nodes["node_name"].astype(str).str.strip()
    return (
        gene_nodes.sort_values("node_index")
        .drop_duplicates("node_name", keep="first")
        .set_index("node_name")
    )


def collect_deg_evidence(
    manifest: pd.DataFrame,
    *,
    deg_dir: Path,
    fdr: float,
    min_abs_log2fc: float,
    direction: str,
    top_genes: int,
) -> pd.DataFrame:
    validate_columns(manifest, MANIFEST_COLUMNS, "Manifest")
    if manifest.empty:
        raise ValueError("Manifest contains no comparisons")

    retained: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        deg_path = deg_dir / row.de_file
        if not deg_path.exists():
            raise FileNotFoundError(deg_path)
        filtered = filter_deg_result(
            pd.read_csv(deg_path),
            fdr=fdr,
            min_abs_log2fc=min_abs_log2fc,
            direction=direction,
            top_genes=top_genes,
        )
        filtered["comparison_id"] = row.comparison_id
        filtered["state_cell_node_name"] = row.state_cell_node_name
        filtered["disease_state"] = row.disease_state
        filtered["cell_type"] = row.cell_type
        retained.append(filtered)

    return pd.concat(retained, ignore_index=True)


def map_evidence_to_genes(retained: pd.DataFrame, gene_map: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    # Keep unmapped counts in the audit instead of silently dropping them.
    mapped = retained.merge(
        gene_map[["node_index", "node_id", "node_source"]],
        left_on="gene_symbol",
        right_index=True,
        how="left",
        indicator=True,
    )
    unmapped_records = int(mapped["_merge"].eq("left_only").sum())
    mapped = mapped.loc[mapped["_merge"].eq("both")].drop(columns="_merge")
    mapped = mapped.rename(
        columns={
            "node_index": "gene_node_index",
            "node_id": "gene_node_id",
            "node_source": "gene_node_source",
        }
    )

    duplicate_mask = mapped.duplicated(["gene_node_index", "state_cell_node_name"], keep="first")
    duplicate_records_removed = int(duplicate_mask.sum())
    evidence = mapped.loc[~duplicate_mask].copy()
    audit = {
        "selected_records_before_mapping": int(len(retained)),
        "unmapped_records": unmapped_records,
        "mapped_records_before_deduplication": int(len(mapped)),
        "duplicate_gene_context_records_removed": duplicate_records_removed,
    }
    return evidence, audit

#  Create new disease-state/cell-type nodes.
def build_state_nodes(nodes: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    # New indices begin after the original graph's highest node index.
    start_index = int(nodes["node_index"].max()) + 1
    contexts = sorted(evidence["state_cell_node_name"].dropna().astype(str).unique())
    return pd.DataFrame(
        {
            "node_index": range(start_index, start_index + len(contexts)),
            "node_id": [f"CRA_STATE_CELL:{name}" for name in contexts],
            "node_type": "cell_type",
            "node_name": contexts,
            "node_source": "CRA_scRNAseq_DESeq2",
        }
    )

def find_cra_node(nodes: pd.DataFrame, cra_node_id: str) -> pd.Series:
    clean_ids = nodes["node_id"].astype(str).str.replace('"', "", regex=False)
    matches = nodes.loc[clean_ids.eq(str(cra_node_id))]
    if matches.empty:
        raise ValueError(f"CRA node_id {cra_node_id!r} was not found in node.csv")
    if len(matches) > 1:
        raise ValueError(f"CRA node_id {cra_node_id!r} is not unique in node.csv")
    return matches.iloc[0]

#Construct the new knowledge-graph edges.
def build_edges(
    evidence: pd.DataFrame,
    state_nodes: pd.DataFrame,
    cra_node: pd.Series,
    *,
    cra_node_id: str,
    cra_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_lookup = state_nodes.set_index("node_name")

    gene_edges = pd.DataFrame(
        {
            "relation": "cell_state_enriched_gene",
            "display_relation": "DESeq2_disease_state_enriched_expression",
            "x_index": evidence["gene_node_index"],
            "x_id": evidence["gene_node_id"],
            "x_type": "gene/protein",
            "x_name": evidence["gene_symbol"],
            "x_source": evidence["gene_node_source"],
            "y_index": evidence["state_cell_node_name"].map(state_lookup["node_index"]),
            "y_id": evidence["state_cell_node_name"].map(state_lookup["node_id"]),
            "y_type": "cell_type",
            "y_name": evidence["state_cell_node_name"],
            "y_source": "CRA_scRNAseq_DESeq2",
        }
    )

    cra_edges = pd.DataFrame(
        {
            "relation": ["CRA_state_cell_context"] * len(state_nodes),
            "display_relation": ["CRA_disease_state_cell_context"] * len(state_nodes),
            "x_index": [int(cra_node["node_index"])] * len(state_nodes),
            "x_id": [cra_node_id] * len(state_nodes),
            "x_type": ["disease"] * len(state_nodes),
            "x_name": [cra_name] * len(state_nodes),
            "x_source": [str(cra_node["node_source"])] * len(state_nodes),
            "y_index": state_nodes["node_index"].tolist(),
            "y_id": state_nodes["node_id"].tolist(),
            "y_type": ["cell_type"] * len(state_nodes),
            "y_name": state_nodes["node_name"].tolist(),
            "y_source": ["CRA_scRNAseq_DESeq2"] * len(state_nodes),
        }
    )
    return gene_edges, cra_edges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deg-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--node-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-genes", type=int, default=250)
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--min-abs-log2fc", type=float, default=0.5)
    parser.add_argument(
        "--direction",
        choices=("disease_enriched", "both"),
        default="disease_enriched",
        help="For NL-minus-AD/SER contrasts, disease_enriched retains negative log2FC genes.",
    )
    parser.add_argument("--cra-node-id", default="5484")
    parser.add_argument("--cra-name", default="colorectal adenoma")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    nodes = pd.read_csv(args.node_csv, sep="\t", low_memory=False)
    validate_columns(nodes, NODE_COLUMNS, "node.csv")
    cra_node = find_cra_node(nodes, args.cra_node_id)

    retained = collect_deg_evidence(
        manifest,
        deg_dir=args.deg_dir,
        fdr=args.fdr,
        min_abs_log2fc=args.min_abs_log2fc,
        direction=args.direction,
        top_genes=args.top_genes,
    )
    evidence, mapping_audit = map_evidence_to_genes(retained, build_gene_map(nodes))
    state_nodes = build_state_nodes(nodes, evidence)
    gene_edges, cra_edges = build_edges(
        evidence,
        state_nodes,
        cra_node,
        cra_node_id=args.cra_node_id,
        cra_name=args.cra_name,
    )

    evidence.to_csv(args.output_dir / "mapped_deg_evidence.csv", index=False)
    state_nodes.to_csv(args.output_dir / "added_state_cell_nodes.tsv", sep="\t", index=False)
    gene_edges.to_csv(args.output_dir / "gene_to_state_cell_edges.csv", index=False)
    cra_edges.to_csv(args.output_dir / "cra_to_state_cell_edges.csv", index=False)

    audit = {
        "selection": {
            "fdr": args.fdr,
            "min_abs_log2fc": args.min_abs_log2fc,
            "direction": args.direction,
            "top_genes_per_comparison": args.top_genes,
        },
        "deg_files_processed": int(len(manifest)),
        **mapping_audit,
        "mapped_gene_state_edges": int(len(gene_edges)),
        "state_cell_nodes": int(len(state_nodes)),
        "cra_context_edges": int(len(cra_edges)),
    }
    (args.output_dir / "evidence_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
