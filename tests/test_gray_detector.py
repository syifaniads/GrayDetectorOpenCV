import numpy as np

from src.gray_detector import GrayRange, apply_mask, build_gray_mask, detect_gray


def pixel(b: int, g: int, r: int) -> np.ndarray:
    return np.array([[[b, g, r]]], dtype=np.uint8)


def test_mid_gray_is_selected() -> None:
    mask = build_gray_mask(pixel(128, 128, 128))
    assert int(mask[0, 0]) == 255


def test_saturated_red_is_rejected() -> None:
    mask = build_gray_mask(pixel(0, 0, 255))
    assert int(mask[0, 0]) == 0


def test_bright_white_exceeds_historical_value_max() -> None:
    mask = build_gray_mask(pixel(255, 255, 255))
    assert int(mask[0, 0]) == 0


def test_detect_gray_returns_mask_and_masked_frame() -> None:
    frame = np.array([[[128, 128, 128], [0, 0, 255]]], dtype=np.uint8)
    mask, result = detect_gray(frame, GrayRange())

    assert mask.tolist() == [[255, 0]]
    assert result[0, 0].tolist() == [128, 128, 128]
    assert result[0, 1].tolist() == [0, 0, 0]


def test_apply_mask_requires_matching_dimensions() -> None:
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.zeros((1, 1), dtype=np.uint8)

    try:
        apply_mask(frame, mask)
    except ValueError as exc:
        assert "dimensions" in str(exc)
    else:
        raise AssertionError("expected dimension validation to fail")
