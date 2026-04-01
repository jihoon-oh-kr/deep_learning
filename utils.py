"""
utils.py
--------
모델 간 연결 및 공통 유틸리티 함수 모음.
Detection → Segmentation → Inpainting → VLM 파이프라인에서 공유하는 함수들.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image


# ──────────────────────────────────────────────
# 데이터 구조
# ──────────────────────────────────────────────

@dataclass
class DetectionResult:
    """단일 객체 탐지 결과."""
    label: str                          # 객체 이름 (예: "cat", "person")
    score: float                        # confidence score (0~1)
    box: Tuple[int, int, int, int]      # (x_min, y_min, x_max, y_max) 픽셀 좌표
    box_id: int = 0                     # UI에서 구분하기 위한 고유 ID


@dataclass
class PipelineState:
    """전체 파이프라인에서 단계 간 상태를 전달하는 컨테이너."""
    original_image: Optional[Image.Image] = None
    detection_results: List[DetectionResult] = field(default_factory=list)
    selected_detection: Optional[DetectionResult] = None  # 사용자가 선택한 박스
    cropped_image: Optional[Image.Image] = None           # 크롭된 원본
    segmentation_mask: Optional[np.ndarray] = None        # 이진 마스크 (H x W)
    background_only: Optional[Image.Image] = None         # 객체 제거 후 배경
    inpainted_image: Optional[Image.Image] = None         # 생성형 모델 결과
    vlm_summary: str = ""                                  # VLM 요약 텍스트


# ──────────────────────────────────────────────
# 바운딩 박스 유틸리티
# ──────────────────────────────────────────────

def crop_image_by_box(
    image: Image.Image,
    box: Tuple[int, int, int, int],
    padding: int = 0,
) -> Image.Image:
    """
    이미지에서 바운딩 박스 영역을 크롭하여 반환.

    Args:
        image:   원본 PIL 이미지
        box:     (x_min, y_min, x_max, y_max)
        padding: 박스 주변에 추가할 여백 (픽셀)

    Returns:
        크롭된 PIL 이미지
    """
    w, h = image.size
    x_min, y_min, x_max, y_max = box
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)
    return image.crop((x_min, y_min, x_max, y_max))


def get_box_info(
    image: Image.Image,
    detection: DetectionResult,
    padding: int = 0,
    save_path: Optional[str] = None,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    사용자가 바운딩 박스 레이블을 클릭했을 때 호출되는 핵심 함수.

    바운딩 박스에 해당하는 크롭 이미지와 좌표를 반환하고,
    선택적으로 크롭 이미지를 파일로 저장합니다.

    Args:
        image:       원본 PIL 이미지
        detection:   클릭된 DetectionResult 객체
        padding:     크롭 시 여백 (픽셀)
        save_path:   저장 경로 (None이면 저장 안 함)

    Returns:
        (cropped_image, box_coords)
        - cropped_image: 바운딩 박스 영역 PIL 이미지
        - box_coords:    (x_min, y_min, x_max, y_max)
    """
    cropped = crop_image_by_box(image, detection.box, padding=padding)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cropped.save(save_path)
        print(f"[utils] 크롭 이미지 저장: {save_path}")

    print(f"[utils] 선택된 객체: '{detection.label}'  "
          f"score={detection.score:.3f}  "
          f"box={detection.box}")

    return cropped, detection.box


def normalize_box(
    box: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> Tuple[float, float, float, float]:
    """
    픽셀 좌표 바운딩 박스를 0~1 범위로 정규화.
    세그멘테이션/인페인팅 모델에 넘길 때 유용.
    """
    x_min, y_min, x_max, y_max = box
    return (
        x_min / image_width,
        y_min / image_height,
        x_max / image_width,
        y_max / image_height,
    )


def scale_box(
    box: Tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """
    정규화된(0~1) 바운딩 박스를 픽셀 좌표로 변환.
    Grounding DINO 등 일부 모델의 출력을 픽셀로 변환할 때 사용.
    """
    x_min, y_min, x_max, y_max = box
    return (
        int(x_min * image_width),
        int(y_min * image_height),
        int(x_max * image_width),
        int(y_max * image_height),
    )


# ──────────────────────────────────────────────
# 이미지 입출력
# ──────────────────────────────────────────────

def load_image(path: str) -> Image.Image:
    """이미지 파일을 RGB PIL 이미지로 로드."""
    img = Image.open(path).convert("RGB")
    print(f"[utils] 이미지 로드: {path}  크기={img.size}")
    return img


def save_image(image: Image.Image, path: str) -> None:
    """PIL 이미지를 파일로 저장."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    image.save(path)
    print(f"[utils] 이미지 저장: {path}")


# ──────────────────────────────────────────────
# 마스크 유틸리티 (세그멘테이션 → 인페인팅 연결용)
# ──────────────────────────────────────────────

def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """
    이진 numpy 마스크 (0/1 또는 0/255) → PIL 'L' 모드 이미지.
    인페인팅 모델 입력 형식에 맞게 변환.
    """
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    return Image.fromarray(mask.astype(np.uint8), mode="L")


def apply_mask_to_image(
    image: Image.Image,
    mask: np.ndarray,
    fill_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    이미지에서 마스크(=1) 영역을 fill_color로 교체.
    세그멘테이션 결과로 객체를 지우고 배경만 남길 때 사용.

    Args:
        image:      원본 PIL 이미지 (RGB)
        mask:       이진 numpy 배열 (H x W), 1=객체, 0=배경
        fill_color: 객체 영역을 채울 색상 (기본: 흰색)

    Returns:
        객체가 fill_color로 교체된 PIL 이미지
    """
    img_array = np.array(image).copy()
    binary = (mask > 0.5).astype(bool)
    img_array[binary] = fill_color
    return Image.fromarray(img_array)


# ──────────────────────────────────────────────
# 파이프라인 단계 연결 헬퍼
# ──────────────────────────────────────────────

def prepare_segmentation_input(
    state: PipelineState,
    padding: int = 10,
) -> Tuple[Optional[Image.Image], Optional[Tuple[int, int, int, int]]]:
    """
    Detection 결과에서 세그멘테이션 모델 입력을 준비.
    선택된 DetectionResult가 없으면 (None, None) 반환.

    Returns:
        (cropped_image, box_coords)
    """
    if state.selected_detection is None or state.original_image is None:
        print("[utils] 선택된 탐지 결과 없음.")
        return None, None

    cropped, box = get_box_info(
        state.original_image,
        state.selected_detection,
        padding=padding,
    )
    state.cropped_image = cropped
    return cropped, box


def prepare_inpainting_input(
    state: PipelineState,
) -> Tuple[Optional[Image.Image], Optional[Image.Image]]:
    """
    Segmentation 결과에서 인페인팅 모델 입력을 준비.

    Returns:
        (background_image, mask_pil)
        - background_image: 객체가 제거된 배경 이미지
        - mask_pil:         인페인팅할 영역을 표시한 마스크 (PIL 'L')
    """
    if state.segmentation_mask is None or state.cropped_image is None:
        print("[utils] 세그멘테이션 마스크 없음.")
        return None, None

    background = apply_mask_to_image(state.cropped_image, state.segmentation_mask)
    mask_pil = mask_to_pil(state.segmentation_mask)
    state.background_only = background
    return background, mask_pil
