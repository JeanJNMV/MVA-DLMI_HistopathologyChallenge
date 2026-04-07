import argparse
import json
from pathlib import Path


ALL_TRAIN_CENTERS = [0, 3, 4]


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-fold LOCO results into a single summary JSON."
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_unfreeze", type=int, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--centers", type=int, nargs="+", default=ALL_TRAIN_CENTERS)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    fold_results = {}

    for center in args.centers:
        result_path = (
            results_dir
            / f"loco_{args.model_name}_{args.num_unfreeze}layers_center{center}.json"
        )
        if not result_path.exists():
            raise FileNotFoundError(f"Missing fold result: {result_path}")

        with result_path.open() as f:
            payload = json.load(f)

        fold_key = f"center_{center}"
        if fold_key not in payload.get("fold_results", {}):
            raise ValueError(f"{result_path} does not contain {fold_key}")
        fold_results[fold_key] = payload["fold_results"][fold_key]

    mean_acc = sum(fold_results.values()) / len(fold_results)
    aggregated = {
        "model_name": args.model_name,
        "num_unfreeze": args.num_unfreeze,
        "fold_results": fold_results,
        "mean_loco_accuracy": mean_acc,
    }

    out_path = results_dir / f"loco_{args.model_name}_{args.num_unfreeze}layers.json"
    with out_path.open("w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Aggregated {len(fold_results)} folds")
    print(f"Mean LOCO accuracy: {mean_acc:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
