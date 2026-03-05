#!/usr/bin/env python3
"""Realtime fire/smoke detection from webcam using YOLO.

Works on CPU by default and uses GPU automatically if available.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
from ultralytics import YOLO


def parse_source(source: str) -> int | str:
    """Return webcam index if numeric, otherwise return path/URL."""
    return int(source) if source.isdigit() else source


def detect_device(preferred: str) -> str:
    """Choose best available device when preferred='auto'."""
    if preferred != "auto":
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def default_model_path(project_root: Path) -> Path:
    """Prefer PT model for better compatibility/performance in Python."""
    pt_model = project_root / "models" / "best.pt"
    tflite_model = project_root / "models" / "best.tflite"
    if pt_model.exists():
        return pt_model
    if tflite_model.exists():
        return tflite_model
    raise FileNotFoundError(
        "No model found. Expected models/best.pt or models/best.tflite."
    )


def normalize_label(label: str) -> str:
    return label.strip().lower()


def load_targets(raw_targets: str) -> List[str]:
    return [normalize_label(x) for x in raw_targets.split(",") if x.strip()]


def model_names_dict(model: YOLO) -> Dict[int, str]:
    names = model.names
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def format_alert_text(labels: Iterable[str]) -> str:
    unique_labels = sorted(set(labels))
    if not unique_labels:
        return "NO ALERT"
    return "ALERT: " + ", ".join(unique_labels).upper()


def draw_status_panel(
    frame,
    alert_text: str,
    fps: float,
    device: str,
    conf: float,
) -> None:
    color = (0, 180, 0) if alert_text == "NO ALERT" else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (20, 20, 20), -1)
    cv2.putText(
        frame,
        alert_text,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f} | DEVICE: {device} | CONF: {conf:.2f}",
        (12, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fire/Smoke realtime detector")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Path to YOLO model (.pt preferred, .tflite optional).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera source index (0,1,...) or stream/video path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda:0 | mps",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--width", type=int, default=960, help="Capture width.")
    parser.add_argument("--height", type=int, default=540, help="Capture height.")
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process 1 frame every N frames (higher = faster on weak PCs).",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="smoke,fire",
        help="Comma-separated labels to track.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional output video path (mp4).",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = Path(args.model) if args.model else default_model_path(project_root)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = detect_device(args.device)
    targets = load_targets(args.targets)
    if not targets:
        raise ValueError("At least one target label is required in --targets.")

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    writer = None
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            20.0,
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )

    print(f"[INFO] Model:  {model_path}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Source: {source}")
    print(f"[INFO] Targets: {targets}")
    print(f"[INFO] Press 'q' to quit.")

    model = YOLO(str(model_path))
    names = model_names_dict(model)

    frame_count = 0
    last_ts = time.time()
    fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] No frame received from source.")
                break

            frame_count += 1
            if args.skip_frames > 1 and (frame_count % args.skip_frames) != 0:
                cv2.imshow("Fire_Smoke_YOLO", frame)
                if writer is not None:
                    writer.write(frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()

            detected_targets: List[str] = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    name = names.get(class_id, str(class_id))
                    normalized_name = normalize_label(name)
                    if normalized_name in targets:
                        detected_targets.append(normalized_name)

            now = time.time()
            dt = max(now - last_ts, 1e-6)
            fps = 1.0 / dt
            last_ts = now

            alert_text = format_alert_text(detected_targets)
            draw_status_panel(annotated, alert_text, fps, device, args.conf)

            cv2.imshow("Fire_Smoke_YOLO", annotated)
            if writer is not None:
                writer.write(annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
