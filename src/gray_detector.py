from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GrayRange:
    """HSV bounds used to classify low-saturation gray pixels."""

    hue_min: int = 0
    saturation_min: int = 0
    value_min: int = 50
    hue_max: int = 180
    saturation_max: int = 50
    value_max: int = 200

    def lower(self) -> np.ndarray:
        return np.array(
            [self.hue_min, self.saturation_min, self.value_min], dtype=np.uint8
        )

    def upper(self) -> np.ndarray:
        return np.array(
            [self.hue_max, self.saturation_max, self.value_max], dtype=np.uint8
        )


def build_gray_mask(frame_bgr: np.ndarray, gray_range: GrayRange = GrayRange()) -> np.ndarray:
    """Return a binary mask for pixels inside the configured HSV gray range."""
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("frame_bgr must contain image data")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a BGR image with shape (H, W, 3)")

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, gray_range.lower(), gray_range.upper())


def apply_mask(frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only pixels selected by a binary mask."""
    if frame_bgr.shape[:2] != mask.shape[:2]:
        raise ValueError("frame and mask dimensions must match")
    return cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)


def detect_gray(
    frame_bgr: np.ndarray, gray_range: GrayRange = GrayRange()
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(mask, masked_frame)` for one BGR frame."""
    mask = build_gray_mask(frame_bgr, gray_range)
    return mask, apply_mask(frame_bgr, mask)


def run_camera(camera_index: int, gray_range: GrayRange) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"unable to open camera index {camera_index}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("camera opened but no frame could be read")

            mask, result = detect_gray(frame, gray_range)
            cv2.imshow("Original", frame)
            cv2.imshow("Gray Mask", mask)
            cv2.imshow("Gray Detection", result)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect low-saturation gray pixels from a camera")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--saturation-max", type=int, default=50, choices=range(0, 256))
    parser.add_argument("--value-min", type=int, default=50, choices=range(0, 256))
    parser.add_argument("--value-max", type=int, default=200, choices=range(0, 256))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.value_min > args.value_max:
        raise SystemExit("--value-min must be <= --value-max")

    gray_range = GrayRange(
        saturation_max=args.saturation_max,
        value_min=args.value_min,
        value_max=args.value_max,
    )
    run_camera(args.camera, gray_range)


if __name__ == "__main__":
    main()
