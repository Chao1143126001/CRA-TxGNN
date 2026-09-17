import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / "scripts" / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_script("prepare_deg_evidence", "01_prepare_deg_evidence.py")
compare = load_script("compare_metrics", "04_compare_metrics.py")


class DegFilteringTest(unittest.TestCase):
    def test_both_direction_ranks_by_absolute_effect_size(self) -> None:
        frame = pd.DataFrame(
            {
                "gene_symbol": ["SMALL_NEG", "LARGE_POS"],
                "p_val_adj": [0.001, 0.001],
                "avg_log2FC": [-0.6, 5.0],
            }
        )
        result = prepare.filter_deg_result(
            frame, fdr=0.05, min_abs_log2fc=0.5, direction="both", top_genes=1
        )
        self.assertEqual(result["gene_symbol"].tolist(), ["LARGE_POS"])

    def test_malformed_deg_table_has_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "adjusted-P column"):
            prepare.filter_deg_result(
                pd.DataFrame({"gene_symbol": ["A"], "avg_log2FC": [1.0]}),
                fdr=0.05,
                min_abs_log2fc=0.5,
                direction="both",
                top_genes=10,
            )


class EvidencePipelineTest(unittest.TestCase):
    def test_evidence_and_graph_augmentation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            base = root / "base"
            deg = root / "deg"
            evidence = root / "evidence"
            augmented = root / "augmented"
            base.mkdir()
            deg.mkdir()

            nodes = pd.DataFrame(
                [
                    [0, "101", "gene/protein", "GENE_A", "NCBI"],
                    [1, "102", "gene/protein", "GENE_B", "NCBI"],
                    [2, "5484", "disease", "colorectal adenoma", "MONDO"],
                ],
                columns=["node_index", "node_id", "node_type", "node_name", "node_source"],
            )
            nodes.to_csv(base / "node.csv", sep="\t", index=False)
            edge_columns = [
                "relation", "display_relation", "x_index", "x_id", "x_type", "x_name", "x_source",
                "y_index", "y_id", "y_type", "y_name", "y_source",
            ]
            pd.DataFrame(
                [["disease_protein", "associated", 2, "5484", "disease", "colorectal adenoma", "MONDO",
                  0, "101", "gene/protein", "GENE_A", "NCBI"]],
                columns=edge_columns,
            ).to_csv(base / "kg.csv", index=False)
            pd.DataFrame(
                {
                    "gene_symbol": ["GENE_A", "GENE_A", "GENE_B", "NOT_IN_KG"],
                    "p_val_adj": [0.001, 0.002, 0.20, 0.001],
                    "avg_log2FC": [-1.2, -1.0, -2.0, -1.0],
                }
            ).to_csv(deg / "NL_ABS_vs_AD_ABS.csv", index=False)
            manifest = pd.DataFrame(
                [{
                    "de_file": "NL_ABS_vs_AD_ABS.csv",
                    "comparison_id": "NL_ABS_vs_AD_ABS",
                    "state_cell_node_name": "AD_ABS",
                    "disease_state": "AD",
                    "cell_type": "ABS",
                }]
            )
            manifest_path = root / "manifest.csv"
            manifest.to_csv(manifest_path, index=False)

            subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY / "scripts" / "01_prepare_deg_evidence.py"),
                    "--deg-dir", str(deg),
                    "--manifest", str(manifest_path),
                    "--node-csv", str(base / "node.csv"),
                    "--output-dir", str(evidence),
                ],
                check=True,
            )
            mapped = pd.read_csv(evidence / "mapped_deg_evidence.csv")
            self.assertEqual(mapped["gene_symbol"].tolist(), ["GENE_A"])
            self.assertEqual(pd.read_csv(evidence / "added_state_cell_nodes.tsv", sep="\t").shape[0], 1)

            audit = json.loads((evidence / "evidence_audit.json").read_text())
            self.assertEqual(audit["unmapped_records"], 1)
            self.assertEqual(audit["duplicate_gene_context_records_removed"], 1)

            subprocess.run(
                [
                    sys.executable,
                    str(REPOSITORY / "scripts" / "02_augment_kg.py"),
                    "--base-kg", str(base),
                    "--evidence-dir", str(evidence),
                    "--output-kg", str(augmented),
                ],
                check=True,
            )
            self.assertEqual(pd.read_csv(base / "kg.csv").shape[0], 1)
            self.assertEqual(pd.read_csv(augmented / "kg.csv").shape[0], 3)
            self.assertEqual(pd.read_csv(augmented / "node.csv", sep="\t").shape[0], 4)

    def test_missing_cra_node_has_clear_error(self) -> None:
        nodes = pd.DataFrame(
            [[0, "101", "gene/protein", "GENE_A", "NCBI"]],
            columns=["node_index", "node_id", "node_type", "node_name", "node_source"],
        )
        with self.assertRaisesRegex(ValueError, "was not found"):
            prepare.find_cra_node(nodes, "5484")


class MatchedComparisonTest(unittest.TestCase):
    def make_results(self, disease_ids: list[str], offset: float = 0.0) -> pd.DataFrame:
        rows = []
        for i, disease_id in enumerate(disease_ids):
            rows.append(
                {
                    "relation": "indication",
                    "disease_id": disease_id,
                    "AUROC": 0.70 + i * 0.01 + offset,
                    "AUPRC": 0.20 + i * 0.01 + offset,
                    "Recall@10": 0.30 + i * 0.01 + offset,
                    "MRR@10": 0.10 + i * 0.01 + offset,
                    "AP@10": 0.15 + i * 0.01 + offset,
                }
            )
        return pd.DataFrame(rows)

    def test_match_runs_pairs_same_diseases(self) -> None:
        original = self.make_results(["d1", "d2"])
        augmented = self.make_results(["d2", "d1"], offset=0.05)
        paired = compare.match_runs(original, augmented)
        self.assertEqual(set(paired["disease_id"]), {"d1", "d2"})
        self.assertEqual(len(paired), 2)

    def test_match_runs_rejects_unmatched_diseases(self) -> None:
        original = self.make_results(["d1", "d2"])
        augmented = self.make_results(["d1", "d3"], offset=0.05)
        with self.assertRaisesRegex(ValueError, "not matched"):
            compare.match_runs(original, augmented)


if __name__ == "__main__":
    unittest.main()
