import argparse
import subprocess
import sys


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the Kaggle-friendly medical CT-CBM pipeline.")
    parser.add_argument("--output-root", default="./output")
    parser.add_argument("--model-name", default="gemma")
    parser.add_argument("--discovery-model", default="google/gemma-2-2b-it")
    parser.add_argument("--concept-clusters", type=int, default=100)
    parser.add_argument("--baseline-epochs", type=int, default=10)
    parser.add_argument("--sample-train", type=int, default=None)
    parser.add_argument("--sample-test", type=int, default=None)
    args = parser.parse_args()

    common = [
        "--dataset", "medical",
        "--model-name", args.model_name,
        "--output-root", args.output_root,
    ]

    annotation = [
        sys.executable, "run_experiments/scripts/run_our_annotation.py",
        *common,
        "--discovery-model", args.discovery_model,
        "--n-cluster", str(args.concept_clusters),
    ]
    baseline = [
        sys.executable, "run_experiments/scripts/run_baseline.py",
        *common,
        "--epochs", str(args.baseline_epochs),
    ]
    cavs = [
        sys.executable, "run_experiments/scripts/run_mean_cavs.py",
        *common,
    ]

    if args.sample_train:
        annotation += ["--sample-train", str(args.sample_train)]
        baseline += ["--sample-train", str(args.sample_train)]
    if args.sample_test:
        annotation += ["--sample-test", str(args.sample_test)]
        baseline += ["--sample-test", str(args.sample_test)]

    run(annotation)
    run(baseline)
    run(cavs)

    print("\nCore pipeline completed: concepts, baseline checkpoint, and mean CAVs are ready.")
    print("You can now run LIG/combined-score/CT-CBM notebooks with annotation='our_annotation'.")


if __name__ == "__main__":
    main()
