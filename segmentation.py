"""
segmentation.py
---------------
SAM2 기반 인터랙티브 세그멘테이션 모듈.

- Detection에서 넘어온 크롭 이미지를 받아 표시
- 좌클릭: 포인트 추가 → 마스크 즉시 업데이트
- 우클릭 버튼: 마지막 포인트 취소
- 확인 버튼: 마스크 확정

모델: facebook/sam2-hiera-tiny  (~1GB RAM)
"""

from __future__ import annotations

import warnings, logging, os
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


import os
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageDraw
from typing import List, Tuple, Optional

from transformers import Sam2Processor, Sam2Model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PipelineState, mask_to_pil, apply_mask_to_image


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

MODEL_ID     = "facebook/sam2-hiera-tiny"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
POINT_RADIUS = 6
MASK_ALPHA   = 140
MASK_COLOR   = (50, 180, 255)
POINT_COLOR  = (255, 60, 60)   # (하위 호환용 — 더 이상 직접 사용 안 함)
POS_COLOR    = (60, 220, 90)   # 포함(+) 포인트 = 초록
NEG_COLOR    = (255, 60, 60)   # 제외(−) 포인트 = 빨강


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────

_processor: Optional[Sam2Processor] = None
_model = None


def load_model():
    global _processor, _model
    if _model is not None:
        return _processor, _model
    _processor = Sam2Processor.from_pretrained(MODEL_ID)
    _model     = Sam2Model.from_pretrained(MODEL_ID).to(DEVICE)
    _model.eval()
    return _processor, _model


# ──────────────────────────────────────────────
# 세그멘테이션
# ──────────────────────────────────────────────

def run_segmentation(
    image: Image.Image,
    points: List[Tuple[int, int]],
    labels: List[int],
) -> np.ndarray:
    processor, model = load_model()

    # SAM2 포인트 형식: [image_level][object_level][point_level][coords]
    # points = [(x0,y0), (x1,y1), ...]  → 각 좌표를 list로 변환해야 4단계가 됨
    # labels = [1, 0, 1, ...]   (1=포함/foreground, 0=제외/background)
    input_points = [[[[x, y] for x, y in points]]]   # (1,1,N,2)
    input_labels = [[list(labels)]]                   # (1,1,N)

    inputs = processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    import torch.nn.functional as F

    # pred_masks shape 출력 (디버그)
    print("[seg] pred_masks shape:", outputs.pred_masks.shape)
    print("[seg] iou_scores shape:", outputs.iou_scores.shape)

    # pred_masks: (batch, num_objects, num_masks, H, W)
    # iou_scores: (batch, num_objects, num_masks)
    pred   = outputs.pred_masks[0, 0]   # (num_masks, H, W)
    scores = outputs.iou_scores[0, 0]   # (num_masks,)

    best_idx  = scores.argmax().item()
    best_mask = pred[best_idx]          # (H, W)

    # 원본 이미지 크기로 리사이즈
    w, h = image.size
    best_mask = best_mask.float().unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    best_mask = F.interpolate(best_mask, size=(h, w), mode="bilinear", align_corners=False)
    best_mask = best_mask.squeeze()     # (H, W)

    mask = (best_mask > 0.0).cpu().numpy().astype(np.uint8)
    return mask


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────

# 세그멘테이션 UI 표시 고정 크기 (이 크기로 이미지를 리사이즈해서 표시)
DISPLAY_SIZE = 512   # 정사각형 기준 최대 크기


def render_overlay(
    image: Image.Image,
    points: List[Tuple[int, int]],
    mask: Optional[np.ndarray] = None,
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """
    이미지를 DISPLAY_SIZE 안에 맞게 리사이즈한 뒤,
    리사이즈된 좌표 기준으로 마스크와 포인트를 그려 반환.
    → Gradio가 이미지를 추가로 늘려도 좌표가 일치함.
    포함(1) 포인트는 초록, 제외(0) 포인트는 빨강으로 표시.
    """
    orig_w, orig_h = image.size

    if labels is None:
        labels = [1] * len(points)

    # 비율 유지하며 DISPLAY_SIZE 안에 맞게 리사이즈
    scale = min(DISPLAY_SIZE / orig_w, DISPLAY_SIZE / orig_h, 1.0)
    disp_w = max(1, int(orig_w * scale))
    disp_h = max(1, int(orig_h * scale))
    base = image.resize((disp_w, disp_h), Image.LANCZOS).convert("RGBA")

    # 마스크도 같은 크기로 리사이즈
    if mask is not None and mask.any():
        mask_resized = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize((disp_w, disp_h), Image.NEAREST)
        mask_alpha   = Image.fromarray((np.array(mask_resized) * (MASK_ALPHA / 255)).astype(np.uint8), mode="L")
        color_layer  = Image.new("RGBA", (disp_w, disp_h), (*MASK_COLOR, MASK_ALPHA))
        overlay      = Image.new("RGBA", (disp_w, disp_h), (0, 0, 0, 0))
        overlay.paste(color_layer, mask=mask_resized)
        base = Image.alpha_composite(base, overlay)

    # 포인트도 scale 적용 — label에 따라 색 구분
    draw = ImageDraw.Draw(base)
    for (x, y), lab in zip(points, labels):
        sx, sy = int(x * scale), int(y * scale)
        r = max(4, int(POINT_RADIUS * scale))
        color = POS_COLOR if lab == 1 else NEG_COLOR
        draw.ellipse([sx-r, sy-r, sx+r, sy+r],
                     fill=(*color, 230),
                     outline=(255, 255, 255, 255),
                     width=2)

    return np.array(base.convert("RGB"))


# ──────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────

_seg_state = {
    "image":        None,   # PIL 이미지
    "points":       [],     # [(x, y), ...]
    "labels":       [],     # [1, 0, ...]  (1=포함, 0=제외)
    "current_mask": None,   # 현재 마스크
    "final_mask":   None,   # 확인 후 최종 마스크
}


def set_seg_image(image: Image.Image):
    """Detection → Segmentation 이미지 전달 시 호출."""
    _seg_state["image"]        = image
    _seg_state["points"]       = []
    _seg_state["labels"]       = []
    _seg_state["current_mask"] = None
    _seg_state["final_mask"]   = None


# ──────────────────────────────────────────────
# Gradio 콜백
# ──────────────────────────────────────────────

def gradio_load_image(image_np):
    """
    단독 실행 시: 사용자가 이미지를 업로드하면 _seg_state에 저장.
    main.py에서는 send_to_segmentation()이 set_seg_image()를 호출하므로 불필요.
    """
    if image_np is None:
        return None, "이미지를 업로드하세요."
    pil = Image.fromarray(image_np).convert("RGB")
    set_seg_image(pil)
    overlay = render_overlay(pil, [], mask=None, labels=[])
    return overlay, "이미지 로드 완료. 클릭해서 포인트를 추가하세요.\n  포함 ➕ 모드 좌클릭: 영역 추가  |  제외 ➖ 모드 좌클릭: 영역 제외  |  '마지막 포인트 취소' 버튼으로 되돌리기"


def gradio_click(mode: str, evt: gr.SelectData):
    """
    좌클릭: 현재 모드(포함/제외)에 따라 포인트 추가 → 즉시 세그멘테이션.
    evt.index는 표시(렌더링)된 이미지 기준 좌표이므로 원본 크기로 변환.
    """
    if _seg_state["image"] is None:
        return None, "먼저 이미지를 로드하세요."

    # 모드 → label (1=포함, 0=제외)
    label = 1 if (mode is not None and str(mode).startswith("포함")) else 0

    # evt.index: render_overlay가 반환한 이미지(DISPLAY_SIZE 기준) 좌표
    disp_x, disp_y = evt.index[0], evt.index[1]

    img_w, img_h = _seg_state["image"].size
    scale = min(DISPLAY_SIZE / img_w, DISPLAY_SIZE / img_h, 1.0)

    # 표시 좌표 → 원본 좌표 역변환
    x = int(disp_x / scale)
    y = int(disp_y / scale)

    # 원본 범위 클램프
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))

    _seg_state["points"].append((x, y))
    _seg_state["labels"].append(label)

    # SAM2는 포함(1) 포인트가 최소 1개 있어야 마스크를 만들 수 있음.
    # 제외 포인트만 있으면 모델을 돌리지 않고 안내만 표시.
    if 1 not in _seg_state["labels"]:
        overlay = render_overlay(
            _seg_state["image"], _seg_state["points"],
            _seg_state["current_mask"], _seg_state["labels"],
        )
        return overlay, (
            "먼저 '포함 ➕' 모드로 대상을 한 번 이상 클릭하세요.\n"
            "(제외 ➖ 포인트만으로는 마스크를 만들 수 없습니다)"
        )

    mask = run_segmentation(
        _seg_state["image"], _seg_state["points"], _seg_state["labels"],
    )
    _seg_state["current_mask"] = mask

    overlay  = render_overlay(
        _seg_state["image"], _seg_state["points"], mask, _seg_state["labels"],
    )
    coverage = int(mask.sum() / mask.size * 100)
    n_pos = _seg_state["labels"].count(1)
    n_neg = _seg_state["labels"].count(0)
    status   = (
        f"포함 {n_pos}개 · 제외 {n_neg}개  |  마스크 커버리지 {coverage}%\n"
        f"제외할 부분이 있으면 '제외 ➖' 모드로 클릭  |  '마지막 포인트 취소'로 되돌리기\n"
        f"만족스러우면 '확인' 버튼을 누르세요."
    )
    return overlay, status


def gradio_undo():
    """마지막 포인트 취소 → 재세그멘테이션."""
    if _seg_state["image"] is None:
        return None, "이미지가 없습니다."

    if not _seg_state["points"]:
        overlay = render_overlay(_seg_state["image"], [], None, [])
        return overlay, "취소할 포인트가 없습니다."

    _seg_state["points"].pop()
    if _seg_state["labels"]:
        _seg_state["labels"].pop()
    points = _seg_state["points"]
    labels = _seg_state["labels"]

    if not points:
        _seg_state["current_mask"] = None
        overlay = render_overlay(_seg_state["image"], [], None, [])
        return overlay, "포인트가 모두 제거됐습니다. 다시 클릭하세요."

    # 포함 포인트가 남아있지 않으면 마스크를 만들 수 없음
    if 1 not in labels:
        _seg_state["current_mask"] = None
        overlay = render_overlay(_seg_state["image"], points, None, labels)
        return overlay, (
            "남은 포인트가 모두 '제외'입니다. '포함 ➕' 포인트를 추가하세요."
        )

    mask = run_segmentation(_seg_state["image"], points, labels)
    _seg_state["current_mask"] = mask
    overlay  = render_overlay(_seg_state["image"], points, mask, labels)
    coverage = int(mask.sum() / mask.size * 100)
    n_pos = labels.count(1)
    n_neg = labels.count(0)
    status   = (
        f"포함 {n_pos}개 · 제외 {n_neg}개  |  마스크 커버리지 {coverage}%\n"
        f"마지막 포인트가 취소됐습니다."
    )
    return overlay, status


def gradio_confirm():
    """확인: 마스크 확정 → 배경 이미지 + 마스크 시각화 반환."""
    if _seg_state["current_mask"] is None:
        return None, None, "마스크가 없습니다. 먼저 이미지를 클릭하세요."

    mask  = _seg_state["current_mask"]
    image = _seg_state["image"]
    _seg_state["final_mask"] = mask

    bg_image  = apply_mask_to_image(image, mask, fill_color=(255, 255, 255))
    mask_vis  = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
    coverage  = int(mask.sum() / mask.size * 100)

    status = (
        f"마스크 확정 완료!\n"
        f"마스크 커버리지: {coverage}%  ({mask.sum():,} / {mask.size:,} px)"
    )
    return np.array(bg_image), np.array(mask_vis), status


def gradio_reset():
    """전체 초기화."""
    _seg_state["points"]        = []
    _seg_state["labels"]        = []
    _seg_state["current_mask"]  = None
    _seg_state["final_mask"]    = None

    if _seg_state["image"] is None:
        return None, "이미지가 없습니다."

    overlay = render_overlay(_seg_state["image"], [], None, [])
    return overlay, "초기화됐습니다. 다시 클릭해서 포인트를 추가하세요."


# ──────────────────────────────────────────────
# UI (단독 실행용)
# ──────────────────────────────────────────────

def build_ui() -> gr.Blocks:

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    * { box-sizing: border-box; }
    body, .gradio-container { background: #080b14 !important; font-family: 'Syne', sans-serif !important; }
    .step-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #3a6f9a; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; }
    .pipeline-badge { display: inline-block; background: #0a1e35; border: 1px solid #1a4060; border-radius: 20px; padding: 3px 12px; font-family: 'Space Mono', monospace; font-size: 0.68rem; color: #4a9fd4; letter-spacing: 0.1em; margin: 2px; }
    .pipeline-badge.active { background: #0d2e50; border-color: #1a6fff; color: #7ecfff; box-shadow: 0 0 8px rgba(30,120,255,0.3); }
    """

    with gr.Blocks(css=css, title="Segmentation Module") as demo:

        gr.HTML("""
        <div style="text-align:center; padding:24px 0 12px 0;">
            <h1 style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;color:#e0f0ff;margin:0;">
                Interactive Segmentation
            </h1>
            <p style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#3a6f9a;
                      text-transform:uppercase;letter-spacing:0.1em;margin:4px 0 0 0;">
                SAM2 · Step 2 of 4
            </p>
            <div style="margin-top:8px;">
                <span class="pipeline-badge">① DETECTION</span>
                <span class="pipeline-badge active">② SEGMENTATION</span>
                <span class="pipeline-badge">③ INPAINTING</span>
                <span class="pipeline-badge">④ VLM SUMMARY</span>
            </div>
        </div>
        """)

        gr.HTML("""
        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#5a9abf;
                    background:#0a1520;border:1px solid #1a3a5c;border-radius:8px;
                    padding:10px 14px;margin-bottom:10px;line-height:1.9;">
            🖱️ <b>포함 ➕ 모드 좌클릭</b> — 영역 추가 &nbsp;|&nbsp;
            🖱️ <b>제외 ➖ 모드 좌클릭</b> — 영역 제외 (예: 셔츠만 남기려면 얼굴·팔을 제외 클릭) &nbsp;|&nbsp;
            ↩️ <b>마지막 포인트 취소</b> &nbsp;|&nbsp;
            ✅ <b>확인</b>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # 단독 실행 시 이미지 업로드 컴포넌트
                upload_image = gr.Image(
                    label="이미지 업로드 (단독 실행 시)",
                    type="numpy",
                    height=160,
                )
                seg_mode = gr.Radio(
                    ["포함 ➕", "제외 ➖"],
                    value="포함 ➕",
                    label="클릭 모드 (좌클릭 시 적용)",
                )
                seg_image = gr.Image(
                    label="클릭해서 세그멘테이션 영역 선택",
                    type="numpy",
                    height=380,
                    interactive=True,
                )
                with gr.Row():
                    confirm_btn = gr.Button("✅  확인 (마스크 확정)", variant="primary")
                    undo_btn    = gr.Button("↩️  마지막 포인트 취소", variant="secondary")
                    reset_btn   = gr.Button("🔄  전체 초기화", variant="secondary")
                seg_status = gr.Textbox(label="상태", lines=3, interactive=False)

            with gr.Column(scale=1):
                result_bg   = gr.Image(label="배경만 남긴 이미지 (객체 제거됨)", type="numpy", height=240)
                result_mask = gr.Image(label="최종 마스크", type="numpy", height=240)

        # 이미지 업로드 → 세그멘테이션 이미지에 표시
        upload_image.change(
            fn=gradio_load_image,
            inputs=[upload_image],
            outputs=[seg_image, seg_status],
        )
        # 좌클릭 (현재 모드 전달)
        seg_image.select(
            fn=gradio_click,
            inputs=[seg_mode],
            outputs=[seg_image, seg_status],
        )
        # 마지막 포인트 취소
        undo_btn.click(fn=gradio_undo, outputs=[seg_image, seg_status])
        # 확인
        confirm_btn.click(fn=gradio_confirm, outputs=[result_bg, result_mask, seg_status])
        # 전체 초기화
        reset_btn.click(fn=gradio_reset, outputs=[seg_image, seg_status])

    return demo


if __name__ == "__main__":
    print("=" * 60)
    print("  Interactive Segmentation — segmentation.py")
    print(f"  모델: {MODEL_ID}")
    print(f"  디바이스: {DEVICE}")
    print("=" * 60)
    load_model()
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7861, show_error=True, inbrowser=True, share=True)