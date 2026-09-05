from __future__ import annotations

import argparse

from src.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the PD24 product model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    create_app(args.model).run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
