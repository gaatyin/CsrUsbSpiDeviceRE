"""
YOLO image classification smoke test (Ultralytics).

Usage (inside D:\\mnist_torch venv):
  python test_classify.py
  python test_classify.py path\\to\\image.jpg
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO classification test")
    parser.add_argument(
        "source",
        nargs="?",
        default="https://ultralytics.com/images/bus.jpg",
        help="Image path or URL (default: Ultralytics sample bus.jpg)",
    )
    parser.add_argument(
        "--model",
        default="yolo11n-cls.pt",
        help="Classification weights (default: yolo11n-cls.pt)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="How many top classes to print (default: 5)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print("[ERROR] Missing dependency:", exc)
        print("Activate your venv and run setup_yolo.bat first.")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] torch={torch.__version__}, device={device}")
    print(f"[INFO] Loading model: {args.model}")

    model = YOLO(args.model)
    results = model.predict(source=args.source, imgsz=224, device=device, verbose=False)

    if not results:
        print("[ERROR] No results returned")
        return 1

    result = results[0]
    if result.probs is None:
        print("[ERROR] Model did not return classification probabilities.")
        print("Make sure you loaded a *-cls.pt model, e.g. yolo11n-cls.pt")
        return 1

    names = result.names
    top_idx = result.probs.top5
    top_conf = result.probs.top5conf.tolist()

    print(f"[INFO] Source: {args.source}")
    print(f"[OK] Top-{min(args.top, len(top_idx))} predictions:")
    for rank, (idx, conf) in enumerate(zip(top_idx, top_conf), start=1):
        if rank > args.top:
            break
        label = names.get(int(idx), str(idx)) if isinstance(names, dict) else names[int(idx)]
        print(f"  {rank}. {label:30s}  conf={conf:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
