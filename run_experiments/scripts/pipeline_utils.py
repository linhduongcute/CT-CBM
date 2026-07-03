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


def configure_paths(config, dataset, output_root=None, save_dir=None):
    if output_root:
        output_root = Path(output_root)
        config.DATASET_PATH = str(output_root / "dataset" / dataset) + "/"
        config.SAVE_PATH = str(output_root / f"results_{dataset}") + "/"
        config.SAVE_PATH_CONCEPTS = str(output_root / f"results_{dataset}" / "concepts_discovery")

    if save_dir:
        config.SAVE_PATH_CONCEPTS = str(Path(save_dir))
        concepts_dir = Path(config.SAVE_PATH_CONCEPTS)
        config.SAVE_PATH = str(concepts_dir.parent) + "/"

    config.annotation = "our_annotation"
    os.makedirs(config.SAVE_PATH_CONCEPTS, exist_ok=True)
    os.makedirs(config.SAVE_PATH, exist_ok=True)
    return config


def maybe_sample_by_label(df, n, seed=42):
    if not n:
        return df
    return df.groupby("label", group_keys=False).apply(
        lambda part: part.sample(min(len(part), n), random_state=seed)
    ).reset_index(drop=True)


def ensure_baseline_checkpoint(config):
    base_dir = Path(config.SAVE_PATH) / "blue_checkpoints" / config.model_name / "BaselineModel"
    classifier_path = base_dir / f"{config.model_name}_classifier_state_dict.pth"
    embedder_path = base_dir / f"{config.model_name}_state_dict.pth"
    if not classifier_path.exists() or not embedder_path.exists():
        raise FileNotFoundError(
            "Missing baseline checkpoint. Run:\n"
            "python run_experiments/scripts/run_baseline.py "
            f"--dataset {config.DATASET} --model-name {config.model_name} --output-root <output_root>"
        )
    return classifier_path, embedder_path
