#!/usr/bin/env python3
"""Realtime fire/smoke detection from webcam using YOLO.

Works on CPU by default and uses GPU automatically if available.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List
from urllib.parse import urlparse

import cv2
import numpy as np
import requests
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


def is_url_source(source: int | str) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def is_stream_url(source: int | str) -> bool:
    if not is_url_source(source):
        return False
    parsed = urlparse(source)
    return "stream" in parsed.path.lower()


def open_cv_capture(source: int | str, width: int, height: int) -> cv2.VideoCapture:
    """Open source with OpenCV using FFmpeg backend for URL streams when possible."""
    if is_url_source(source):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def mjpeg_frame_generator(url: str, timeout_sec: float) -> Iterator[np.ndarray]:
    """Read MJPEG bytes directly from HTTP stream and decode JPEG frames."""
    with requests.get(url, stream=True, timeout=timeout_sec) as response:
        response.raise_for_status()
        buffer = b""
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            buffer += chunk

            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9")

            while start != -1 and end != -1 and end > start:
                jpg = buffer[start : end + 2]
                buffer = buffer[end + 2 :]

                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    yield frame

                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9")


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
    parser.add_argument(
        "--stream-timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for MJPEG streams.",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=1.5,
        help="Delay in seconds before retrying stream connection.",
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
    use_mjpeg = is_stream_url(source)
    cap = None
    mjpeg_iter = None

    writer = None
    out_path = Path(args.save) if args.save else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Model:  {model_path}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Source: {source}")
    print(f"[INFO] Targets: {targets}")
    if use_mjpeg:
        print("[INFO] Stream source detected. Using MJPEG HTTP reader with auto-reconnect.")
    else:
        cap = open_cv_capture(source, args.width, args.height)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")
    print(f"[INFO] Press 'q' to quit.")

    model = YOLO(str(model_path))
    names = model_names_dict(model)

    frame_count = 0
    last_ts = time.time()
    fps = 0.0

    try:
        while True:
            if use_mjpeg:
                if mjpeg_iter is None:
                    try:
                        mjpeg_iter = mjpeg_frame_generator(str(source), args.stream_timeout)
                    except Exception as exc:
                        print(
                            f"[WARN] Stream connection failed: {exc}. "
                            f"Retrying in {args.reconnect_delay:.1f}s..."
                        )
                        time.sleep(args.reconnect_delay)
                        continue

                try:
                    frame = next(mjpeg_iter)
                except Exception as exc:
                    print(
                        f"[WARN] Stream read failed: {exc}. "
                        f"Reconnecting in {args.reconnect_delay:.1f}s..."
                    )
                    mjpeg_iter = None
                    time.sleep(args.reconnect_delay)
                    continue
            else:
                assert cap is not None
                ok, frame = cap.read()
                if not ok:
                    print("[WARN] No frame received from source.")
                    break

            if writer is None and out_path is not None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(out_path),
                    fourcc,
                    20.0,
                    (frame.shape[1], frame.shape[0]),
                )

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
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
