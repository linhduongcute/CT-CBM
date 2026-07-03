import argparse

from pipeline_utils import configure_paths, maybe_sample_by_label
from load_config import load_config
from prepare_data import load_fc_prepare_data
from models.utils import load_model_and_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train and save the black-box baseline checkpoint.")
    parser.add_argument("--dataset", default="medical")
    parser.add_argument("--model-name", default="gemma")
    parser.add_argument("--output-root", default="./output")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--sample-train", type=int, default=None)
    parser.add_argument("--sample-test", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.model_name, args.dataset)
    configure_paths(config, args.dataset, output_root=args.output_root)
    if args.epochs is not None:
        config.num_epochs = args.epochs

    prepare_data = load_fc_prepare_data(args.dataset)
    train_loader, test_loader, val_loader, train_df, val_df, test_df = prepare_data(config)

    if args.sample_train or args.sample_test:
        from concepts_bank_utils import create_dataloader

        _, tokenizer, _, _ = load_model_and_tokenizer(config)
        train_df = maybe_sample_by_label(train_df, args.sample_train, config.seed)
        test_df = maybe_sample_by_label(test_df, args.sample_test, config.seed)
        train_loader = create_dataloader(train_df, tokenizer, config.max_len, config.batch_size)
        test_loader = create_dataloader(test_df, tokenizer, config.max_len, config.batch_size)

    if config.model_name == "gemma":
        from models.BaselineModel_gemma import BaselineModel
    else:
        from models.BaselineModel import BaselineModel

    embedder_model, _, _, classifier = load_model_and_tokenizer(config)
    baseline = BaselineModel(
        embedder_model,
        classifier,
        train_loader,
        val_loader,
        test_loader,
        config,
        save_path=config.SAVE_PATH,
        use_cls_token=config.use_cls_token,
    )
    baseline.train_model()
    baseline.load_model()
    baseline.evaluate_model(test_loader, "Test")
    baseline.save_performance_json()


if __name__ == "__main__":
    main()
