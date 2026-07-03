import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([
    str(ROOT / "run_experiments"),
    str(ROOT / "run_experiments" / "scripts"),
    str(ROOT / "run_experiments" / "models"),
    str(ROOT / "run_experiments" / "data"),
])

from annotation_ours import launch_our_annotation
from load_config import load_config
from prepare_data import load_fc_prepare_data


def main():
    parser = argparse.ArgumentParser(description="Generate concepts automatically with an LLM.")
    parser.add_argument("--dataset", default="medical")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--discovery-model", default="google/gemma-2-2b-it")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--n-cluster", type=int, default=100)
    parser.add_argument("--sample-train", type=int, default=None)
    parser.add_argument("--sample-test", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.model_name, args.dataset)

    if args.output_root:
        output_root = Path(args.output_root)
        config.DATASET_PATH = str(output_root / "dataset" / args.dataset)
        config.SAVE_PATH = str(output_root / f"results_{args.dataset}") + "/"
        config.SAVE_PATH_CONCEPTS = str(output_root / f"results_{args.dataset}" / "concepts_discovery")

    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = config.SAVE_PATH_CONCEPTS

    save_dir = str(Path(save_dir))
    os.makedirs(save_dir, exist_ok=True)
    config.SAVE_PATH_CONCEPTS = save_dir

    prepare_data = load_fc_prepare_data(args.dataset)
    train_loader, test_loader, val_loader, train_df, val_df, test_df = prepare_data(config)

    if args.sample_train:
        train_df = train_df.groupby("label", group_keys=False).apply(
            lambda df: df.sample(min(len(df), args.sample_train), random_state=config.seed)
        )
    if args.sample_test:
        test_df = test_df.groupby("label", group_keys=False).apply(
            lambda df: df.sample(min(len(df), args.sample_test), random_state=config.seed)
        )

    launch_our_annotation(
        model_name=args.discovery_model,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config,
        save_dir=save_dir,
        n_cluster=args.n_cluster,
    )


if __name__ == "__main__":
    main()
