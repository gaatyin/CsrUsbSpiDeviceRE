"""
YOLO real-time object detection from local webcam (Ultralytics).

Usage (inside D:\\mnist_torch venv):
  python test_detect_camera.py
  python test_detect_camera.py --camera 0
  python test_detect_camera.py --model yolo11n.pt --conf 0.4

Press q in the preview window to quit.
"""

from __future__ import annotations

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO webcam real-time detection")
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Detection weights (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cpu / 0 (default: auto)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import cv2
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print("[ERROR] Missing dependency:", exc)
        print("Activate your venv and run setup_yolo.bat first.")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] torch={torch.__version__}, device={device}")
    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    # CAP_DSHOW helps OpenCV open webcams more reliably on Windows.
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(args.camera)
    else:
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {args.camera}")
        print("Try another index, e.g. --camera 1")
        return 1

    window = "YOLO Detect (press q to quit)"
    print(f"[INFO] Camera {args.camera} opened. Press q to quit.")
    frame_count = 0
    fps = 0.0
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERROR] Failed to read frame from camera")
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )
            annotated = results[0].plot()

            frame_count += 1
            elapsed = time.perf_counter() - t0
            fps = frame_count / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[OK] Stopped. Processed {frame_count} frames, avg FPS={fps:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
