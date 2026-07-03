import argparse
import json
from pathlib import Path

import torch

from pipeline_utils import configure_paths, ensure_baseline_checkpoint
from load_config import load_config
from models.utils import load_model_and_tokenizer
from prepare_data import prepare_data_from_csv


def main():
    parser = argparse.ArgumentParser(description="Create mean-minus-other CAVs from generated concepts.")
    parser.add_argument("--dataset", default="medical")
    parser.add_argument("--model-name", default="gemma")
    parser.add_argument("--output-root", default="./output")
    args = parser.parse_args()

    config = load_config(args.model_name, args.dataset)
    configure_paths(config, args.dataset, output_root=args.output_root)
    config.cavs_type = "mean"
    config.cavs_type_arg = "mean"

    ensure_baseline_checkpoint(config)

    if config.model_name == "gemma":
        from models.BaselineModel_gemma import BaselineModel
        from mean_cavs_creation_gemma import compute_cavs_mean_minus
    else:
        from models.BaselineModel import BaselineModel
        from mean_cavs_creation import compute_cavs_mean_minus

    embedder_model, embedder_tokenizer, _, classifier = load_model_and_tokenizer(config)
    baseline = BaselineModel(
        embedder_model,
        classifier,
        None,
        None,
        None,
        config,
        save_path=config.SAVE_PATH,
        use_cls_token=config.use_cls_token,
    )
    baseline.load_model()

    df_aug_train = prepare_data_from_csv(annotation="our_annotation", config=config)
    cavs = compute_cavs_mean_minus(
        df_aug_train,
        embedder_model,
        embedder_tokenizer,
        baseline,
        config,
    )

    cavs_dir = Path(config.SAVE_PATH) / "blue_checkpoints" / config.model_name / "cavs" / config.cavs_type
    cavs_dir.mkdir(parents=True, exist_ok=True)

    concept_frequency = {}
    for concept in cavs:
        column = f"dummy_{concept}"
        if column in df_aug_train:
            concept_frequency[concept] = int(df_aug_train[column].sum())
        else:
            concept_frequency[concept] = 0

    sorted_macro_concepts = dict(sorted(concept_frequency.items(), key=lambda item: item[1], reverse=True))
    with open(cavs_dir / "sorted_macro_concepts.json", "w", encoding="utf-8") as f:
        json.dump(sorted_macro_concepts, f, ensure_ascii=False, indent=2)

    # Some downstream notebooks look for this frequency file.
    with open(Path(config.SAVE_PATH_CONCEPTS) / "sorted_macro_concepts_freq.json", "w", encoding="utf-8") as f:
        json.dump(sorted_macro_concepts, f, ensure_ascii=False, indent=2)

    torch.cuda.empty_cache()
    print(f"CAV files saved to {cavs_dir}")


if __name__ == "__main__":
    main()
