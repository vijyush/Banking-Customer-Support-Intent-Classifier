from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from banking_intent.predict import predict_intent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a BANKING77 customer message")
    parser.add_argument("text", help="Customer-support message")
    parser.add_argument(
        "--model", choices=["baseline", "transformer"], default="transformer"
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(predict_intent(args.text, args.model, args.top_k), indent=2))


if __name__ == "__main__":
    main()

