"""
detection.py
------------
Grounding DINO (tiny) 기반 Open-Vocabulary 객체 탐지 모듈.

사용법:
    python detection.py

모델: IDEA-Research/grounding-dino-tiny
  - RAM ~1.5 GB (CPU 추론 기준), GPU 사용 시 ~700 MB VRAM
  - 텍스트 프롬프트로 임의의 객체 탐지 (학습 없이 사용 가능)
  - HuggingFace transformers 라이브러리로 바로 로드

의존성 설치:
    pip install transformers torch torchvision pillow gradio numpy
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
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DetectionResult, PipelineState, get_box_info, load_image, save_image


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BOX_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
    "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
    "#BB8FCE", "#85C1E9", "#82E0AA", "#F0B27A",
]


# ──────────────────────────────────────────────
# 모델 로드 (전역 캐시)
# ──────────────────────────────────────────────

_processor: Optional[AutoProcessor] = None
_model = None


def load_model():
    global _processor, _model
    if _model is not None:
        return _processor, _model
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = AutoModelForZeroShotObjectDetection.from_pretrained(
        MODEL_ID, token=False,
    ).to(DEVICE)
    _model.eval()
    return _processor, _model


# ──────────────────────────────────────────────
# 탐지 핵심 함수
# ──────────────────────────────────────────────

def run_detection(image: Image.Image, text_prompt: str) -> List[DetectionResult]:
    processor, model = load_model()

    prompt = text_prompt.strip()
    if not prompt.endswith("."):
        prompt += " ."

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    w, h = image.size
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        target_sizes=[(h, w)],
    )[0]

    detections: List[DetectionResult] = []
    for idx, (box, score, label) in enumerate(
        zip(results["boxes"], results["scores"], results["labels"])
    ):
        x_min, y_min, x_max, y_max = box.cpu().int().tolist()
        detections.append(DetectionResult(
            label=label,
            score=float(score),
            box=(x_min, y_min, x_max, y_max),
            box_id=idx,
        ))

    print(f"[detection] 탐지된 객체: {len(detections)}개")
    for d in detections:
        print(f"  [{d.box_id}] {d.label}  score={d.score:.3f}  box={d.box}")

    return detections


# ──────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────

def draw_detections(image: Image.Image, detections: List[DetectionResult]) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size

    font_size = max(14, int(min(w, h) * 0.025))
    font = None
    for font_path in [
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, font_size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    for det in detections:
        color = BOX_COLORS[det.box_id % len(BOX_COLORS)]
        x_min, y_min, x_max, y_max = det.box
        line_width = max(2, int(min(w, h) * 0.004))

        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=line_width)

        label_text = f"[{det.box_id}] {det.label} {det.score:.2f}"
        bbox_text = draw.textbbox((0, 0), label_text, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]
        padding = 4

        bg_y0 = max(0, y_min - text_h - padding * 2)
        bg_y1 = max(text_h + padding * 2, y_min)
        draw.rectangle([x_min, bg_y0, x_min + text_w + padding * 2, bg_y1], fill=color)
        draw.text((x_min + padding, bg_y0 + padding), label_text, fill="white", font=font)

    return img


# ──────────────────────────────────────────────
# 박스 선택 함수 (utils.get_box_info 활용)
# ──────────────────────────────────────────────

def on_label_click(
    original_image: Image.Image,
    detections: List[DetectionResult],
    selected_box_id: int,
    padding: int = 10,
) -> Tuple[Image.Image, str]:
    selected = next((d for d in detections if d.box_id == selected_box_id), None)
    if selected is None:
        blank = Image.new("RGB", (200, 200), "#1a1a2e")
        return blank, "선택된 박스를 찾을 수 없습니다."

    cropped, box_coords = get_box_info(original_image, selected, padding=padding)

    x_min, y_min, x_max, y_max = box_coords
    box_w = x_max - x_min
    box_h = y_max - y_min

    info_text = (
        f"객체: {selected.label}\n"
        f"신뢰도: {selected.score:.4f}\n"
        f"바운딩 박스 좌표:\n"
        f"  x_min={x_min},  y_min={y_min}\n"
        f"  x_max={x_max},  y_max={y_max}\n"
        f"크기: {box_w} x {box_h} px\n"
        f"박스 ID: {selected.box_id}"
    )

    return cropped, info_text


# ──────────────────────────────────────────────
# Gradio 콜백
# ──────────────────────────────────────────────

_state = PipelineState()
_current_detections: List[DetectionResult] = []


def gradio_detect(image_input, text_prompt: str):
    global _state, _current_detections

    if image_input is None:
        return None, "이미지를 업로드하세요.", gr.Dropdown(choices=[]), ""

    if not text_prompt.strip():
        return image_input, "탐지할 객체를 입력하세요.\n예)  cat . dog . person . car", gr.Dropdown(choices=[]), ""

    pil_image = Image.fromarray(image_input).convert("RGB")
    _state.original_image = pil_image

    detections = run_detection(pil_image, text_prompt)
    _current_detections = detections
    _state.detection_results = detections

    if not detections:
        return (
            image_input,
            "탐지된 객체가 없습니다. 다른 객체 이름을 입력해보세요.",
            gr.Dropdown(choices=[]),
            "",
        )

    annotated = draw_detections(pil_image, detections)
    choices = [f"[{d.box_id}] {d.label} ({d.score:.2f})" for d in detections]

    summary = (
        f"탐지 완료: {len(detections)}개 객체 발견\n"
        + "\n".join(
            f"  [{d.box_id}] {d.label}  score={d.score:.3f}  box={d.box}"
            for d in detections
        )
    )

    return (
        np.array(annotated),
        summary,
        gr.Dropdown(choices=choices, value=choices[0], label="객체 선택"),
        "객체를 선택하면 크롭 이미지와 좌표가 여기 표시됩니다.",
    )


def gradio_select_box(selected_label_str: str):
    global _state, _current_detections

    if not selected_label_str or not _current_detections or _state.original_image is None:
        blank = Image.new("RGB", (300, 300), "#0d0d1a")
        return np.array(blank), "먼저 탐지를 실행하세요."

    try:
        box_id = int(selected_label_str.split("]")[0].replace("[", "").strip())
    except (ValueError, IndexError):
        blank = Image.new("RGB", (300, 300), "#0d0d1a")
        return np.array(blank), "선택 파싱 오류."

    cropped, info_text = on_label_click(
        _state.original_image,
        _current_detections,
        box_id,
        padding=10,
    )
    _state.selected_detection = next(
        (d for d in _current_detections if d.box_id == box_id), None
    )
    _state.cropped_image = cropped   # ← 전달을 위해 저장
    return np.array(cropped), info_text


# ──────────────────────────────────────────────
# Gradio UI 빌드
# ──────────────────────────────────────────────

def build_ui() -> gr.Blocks:

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

    * { box-sizing: border-box; }

    body, .gradio-container {
        background: #080b14 !important;
        font-family: 'Syne', sans-serif !important;
    }

    h1.title {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.2rem;
        color: #e0f0ff;
        letter-spacing: -0.02em;
        margin-bottom: 4px;
        text-shadow: 0 0 30px rgba(100,180,255,0.3);
    }

    .subtitle {
        font-family: 'Space Mono', monospace;
        font-size: 0.78rem;
        color: #4a7fa5;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 24px;
    }

    .step-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        color: #3a6f9a;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .pipeline-badge {
        display: inline-block;
        background: #0a1e35;
        border: 1px solid #1a4060;
        border-radius: 20px;
        padding: 3px 12px;
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        color: #4a9fd4;
        letter-spacing: 0.1em;
        margin: 2px;
    }

    .pipeline-badge.active {
        background: #0d2e50;
        border-color: #1a6fff;
        color: #7ecfff;
        box-shadow: 0 0 8px rgba(30,120,255,0.3);
    }
    """

    with gr.Blocks(css=css, title="Open-Vocab Detection Pipeline") as demo:

        gr.HTML("""
        <div style="text-align:center; padding: 28px 0 16px 0;">
            <h1 class="title">Open-Vocabulary Object Detection</h1>
            <p class="subtitle">Grounding DINO · Step 1 of 4 · Detection Module</p>
            <div style="margin-top:10px;">
                <span class="pipeline-badge active">① DETECTION</span>
                <span class="pipeline-badge">② SEGMENTATION</span>
                <span class="pipeline-badge">③ INPAINTING</span>
                <span class="pipeline-badge">④ VLM SUMMARY</span>
            </div>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<p class="step-label">Input</p>')
                image_input = gr.Image(label="이미지 업로드", type="numpy", height=340)
                text_prompt = gr.Textbox(
                    label="탐지할 객체 입력  ('. '으로 구분 — 원하는 것 무엇이든 입력 가능)",
                    value="",
                    lines=2,
                    placeholder="예)  cat . dog . person . car . bicycle . laptop",
                )
                detect_btn = gr.Button("탐지 실행", variant="primary")

            with gr.Column(scale=1):
                gr.HTML('<p class="step-label">Detection Result</p>')
                annotated_output = gr.Image(label="탐지 결과", type="numpy", height=340)
                detection_summary = gr.Textbox(
                    label="탐지 요약",
                    lines=5,
                    interactive=False,
                )

        gr.HTML('<hr style="border-color:#1a2a3a; margin: 20px 0;">')

        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML('<p class="step-label">Select Object → Next Step</p>')
                box_selector = gr.Dropdown(
                    label="탐지된 객체 목록 — 선택하면 크롭 이미지와 좌표 출력",
                    choices=[],
                    interactive=True,
                )
                box_info = gr.Textbox(
                    label="바운딩 박스 좌표 정보",
                    lines=8,
                    interactive=False,
                )

            with gr.Column(scale=1):
                gr.HTML('<p class="step-label">Cropped Region</p>')
                cropped_output = gr.Image(
                    label="선택된 객체 크롭 (세그멘테이션 모델 입력)",
                    type="numpy",
                    height=320,
                )

        gr.HTML("""
        <div style="text-align:center; padding: 16px 0 8px 0;
                    font-family:'Space Mono',monospace; font-size:0.7rem;
                    color:#2a4a6a; letter-spacing:0.1em;">
            GROUNDING DINO TINY · HUGGINGFACE TRANSFORMERS · 12GB RAM COMPATIBLE
        </div>
        """)

        detect_btn.click(
            fn=gradio_detect,
            inputs=[image_input, text_prompt],
            outputs=[annotated_output, detection_summary, box_selector, box_info],
        )

        box_selector.change(
            fn=gradio_select_box,
            inputs=[box_selector],
            outputs=[cropped_output, box_info],
        )

    return demo


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Open-Vocabulary Detection — detection.py")
    print(f"  모델: {MODEL_ID}")
    print(f"  디바이스: {DEVICE}")
    print("=" * 60)

    load_model()

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,
    )