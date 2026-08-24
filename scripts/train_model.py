from __future__ import annotations

import argparse

from src.product import save_bundle, train_product


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the five-variable PD product model.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle = train_product(args.data)
    save_bundle(bundle, args.output)
    print(
        {
            "model_version": bundle["model_version"],
            "selected_model_name": bundle["selected_model_name"],
            "data_sha256": bundle["data_sha256"],
            "validation_metrics": bundle["validation_metrics"],
        }
    )


if __name__ == "__main__":
    main()
