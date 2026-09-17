import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from txgnn import TxData, TxEval, TxGNN


DISEASE_ID_CANDIDATES = ("disease_id", "disease_idx", "disease_index", "disease_node_index")


# Define all run settings exposed at the command line.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0", help="Use cpu when no compatible GPU is available.")
    parser.add_argument("--split", default="complex_disease")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pretrain-epochs", type=int, default=1)
    parser.add_argument("--finetune-epochs", type=int, default=500)
    parser.add_argument("--pretrain-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()

#Seed Python, NumPy, and PyTorch for repeatable experiments.
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_disease_identifier(frame: pd.DataFrame) -> pd.DataFrame:
    exported = frame.copy()
    found = next((name for name in DISEASE_ID_CANDIDATES if name in exported.columns), None)
    if found is not None:
        if found != "disease_id":
            exported = exported.rename(columns={found: "disease_id"})
        return exported

    if exported.index.name and "disease" in str(exported.index.name).lower():
        return exported.reset_index().rename(columns={exported.index.name: "disease_id"})

    if not isinstance(exported.index, pd.RangeIndex):
        exported.insert(0, "disease_id", exported.index.astype(str))
        return exported.reset_index(drop=True)

    raise ValueError(
        "TxEval returned disease-level metrics without a stable disease identifier. "
        "A disease_id/disease_idx column or disease-valued index is required for a matched comparison."
    )

def main() -> None:
    args = parse_args()

    # Validate the KG and create a reproducible run directory.
    if not (args.data_folder / "kg.csv").exists() or not (args.data_folder / "node.csv").exists():
        raise FileNotFoundError("--data-folder must contain kg.csv and node.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    # Load the KG and create the disease-held-out split.
    data = TxData(data_folder_path=str(args.data_folder))
    data.prepare_split(split=args.split, seed=args.seed, no_kg=False)

    # Construct the TxGNN training object.
    model = TxGNN(
        data=data, weight_bias_track=False, proj_name="CRA_TxGNN",
        exp_name=args.run_name, device=args.device,
    )

    # Initialize the model architecture.
    model.model_initialize(
        n_hid=args.hidden_dim, n_inp=args.hidden_dim, n_out=args.hidden_dim,
        proto=True, proto_num=5, attention=False, sim_measure="all_nodes_profile",
        bert_measure="disease_name", agg_measure="rarity", exp_lambda=0.7,
        num_walks=200, walk_mode="bit", path_length=2,
    )

    
    # Pretrain on all KG relations, then fine-tune treatment relations.  
    model.pretrain(
        n_epoch=args.pretrain_epochs, learning_rate=args.pretrain_lr,
        batch_size=args.batch_size, train_print_per_n=20,
    )
    model.finetune(
        n_epoch=args.finetune_epochs, learning_rate=args.finetune_lr,
        train_print_per_n=5, valid_per_n=20,
        save_name=str(args.output_dir / "finetune_log"),
    )
    model_dir = args.output_dir / "model"
    model.save_model(str(model_dir))

    # Evaluate drug rankings for every held-out test disease.
    evaluator = TxEval(model=model)
    disease_results = evaluator.eval_disease_centric(
        disease_idxs="test_set", show_plot=False, verbose=True,
        save_result=True, return_raw=False,
        save_name=str(args.output_dir / "disease_centric_evaluation.pkl"),
    )
    tidy_results = []
    for relation, frame in disease_results.items():
        exported = ensure_disease_identifier(frame)
        exported.insert(0, "relation", relation.removeprefix("rev_"))
        tidy_results.append(exported)
    pd.concat(tidy_results, ignore_index=True).to_csv(
        args.output_dir / "per_disease_metrics.csv", index=False
    )
    run_config = vars(args) | {"data_folder": str(args.data_folder), "model_dir": str(model_dir)}
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str))
    print(f"COMPLETE: {args.output_dir}")


if __name__ == "__main__":
    main()
