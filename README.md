# CRA-augmented TxGNN

This is a code sample from my colorectal adenoma (CRA) drug-repurposing project. I wanted to test whether cell-state information from single-cell RNA-seq could add useful context to the TxGNN knowledge graph.

The workflow filters differential-expression results, maps genes to existing knowledge-graph nodes, incorporates CRA state- and cell-type-specific relationships, and compares the original and augmented graphs under the same training settings.

The repository contains code only. The scRNA-seq data, DEG tables, knowledge graph, model weights, and study metadata are not included.

## Approach

For each matched cell type, I start with a normal-versus-AD/SER differential-expression result. A gene is retained when:

- adjusted P value < 0.05;
- |log2 fold change| >= 0.5;
- its symbol maps to a TxGNN gene/protein node.

By default, the code keeps the disease-enriched side of an `NL - AD/SER` contrast and takes at most 250 genes per comparison. I used the cap to prevent a small number of DEG-rich cell states from dominating the added graph. Both the direction and cap are command-line options.

The added path has this form:

```text
drug -> target gene/protein -> CRA state/cell context -> colorectal adenoma
```

The original knowledge graph is preserved unchanged. Before adding new nodes and edges, `02_augment_kg.py` creates a copy of the graph in a separate output directory.

## Files

- `01_prepare_deg_evidence.py`: filters DEG tables and creates TxGNN-compatible nodes and edges.
- `02_augment_kg.py`: makes a separate augmented copy of the graph.
- `03_train_evaluate.py`: trains TxGNN and exports disease-level evaluation results.
- `04_compare_metrics.py`: checks that both runs contain the same held-out diseases, then compares AUROC, AUPRC, Recall@10, MRR@10, and AP@10.
- `tests/test_pipeline.py`: small tests using synthetic data only.

## Local data layout

```text
data/
  base_kg/
    kg.csv
    node.csv
  deseq2_results/
metadata/
  deg_manifest.csv
```

The folders are present in the repository, but their contents are ignored by Git. The private manifest must contain:

```text
de_file,comparison_id,state_cell_node_name,disease_state,cell_type
```

Each DEG CSV needs a gene-symbol column, an adjusted-P-value column, and a log2-fold-change column. The loader accepts common names such as `gene_symbol`, `p_val_adj`, and `avg_log2FC`.

## Environment

I ran this workflow in a Linux GPU environment. TxGNN has specific PyTorch/DGL compatibility requirements, so I install those first and then install the official TxGNN source:

```bash
pip install -r requirements.txt
git clone https://github.com/mims-harvard/TxGNN.git external/TxGNN
pip install -e external/TxGNN
```

## Running the workflow

Prepare the graph evidence:

```bash
python scripts/01_prepare_deg_evidence.py \
  --deg-dir data/deseq2_results \
  --manifest metadata/deg_manifest.csv \
  --node-csv data/base_kg/node.csv \
  --output-dir outputs/evidence \
  --top-genes 250 \
  --direction disease_enriched
```

Build the augmented graph:

```bash
python scripts/02_augment_kg.py \
  --base-kg data/base_kg \
  --evidence-dir outputs/evidence \
  --output-kg data/CRA_augmented_KG
```

Train the baseline and augmented models with matched settings:

```bash
python scripts/03_train_evaluate.py \
  --data-folder data/base_kg \
  --output-dir outputs/original_seed43 \
  --run-name original_seed43 \
  --seed 43 --device cuda:0

python scripts/03_train_evaluate.py \
  --data-folder data/CRA_augmented_KG \
  --output-dir outputs/cra_seed43 \
  --run-name cra_seed43 \
  --seed 43 --device cuda:0
```

Compare the disease-level results:

```bash
python scripts/04_compare_metrics.py \
  --original outputs/original_seed43/per_disease_metrics.csv \
  --augmented outputs/cra_seed43/per_disease_metrics.csv \
  --output-dir outputs/comparison_seed43 \
  --title "Seed 43: Original versus CRA-augmented TxGNN"
```

Run the lightweight tests:

```bash
python -m unittest discover -s tests -v
```

## Evaluation notes

The comparison is paired by disease ID and treatment relation. I keep the split, seed, architecture, and training settings the same so that graph content is the intended experimental difference.

AUPRC is particularly important here because known drug-disease links are sparse. I still report AUROC and the top-10 ranking metrics because they describe different aspects of model behavior. An improvement in one metric or relation does not imply that the augmented model is better everywhere.

## Limitations

- Differential expression is association evidence, not a causal drug mechanism.
- Model rankings are research hypotheses, not clinical recommendations.

## TxGNN citation

Huang K, et al. *A foundation model for clinician-centered drug repurposing.* Nature Medicine (2024). https://doi.org/10.1038/s41591-024-03233-x

Please retain the original TxGNN license and citation when reusing its implementation.
