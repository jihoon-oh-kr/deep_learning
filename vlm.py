"""
vlm.py
------
Qwen2.5-VL 기반 파이프라인 요약 모듈.

입력:
  - original_image:  원본 전체 이미지 (PIL)
  - final_image:     인페인팅 후 합성된 전체 이미지 (PIL)
  - label:           제거된 객체 이름 (detection 결과)
  - user_question:   사용자 추가 질문 (선택)

출력:
  - 파이프라인 과정 요약 텍스트

모델: Qwen/Qwen2.5-VL-3B-Instruct
  - RAM ~6GB, VRAM ~3GB
  - 멀티 이미지 입력 지원
  - HuggingFace transformers로 바로 로드
"""

from __future__ import annotations

import os
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PipelineState


# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────

_model     = None
_processor = None


def load_model():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"[vlm] 모델 로드 중: {MODEL_ID}  (device={DEVICE})")
    _processor = AutoProcessor.from_pretrained(MODEL_ID)
    _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )
    _model.eval()
    print("[vlm] 모델 로드 완료.")
    return _model, _processor


# ──────────────────────────────────────────────
# 프롬프트 생성
# ──────────────────────────────────────────────

def build_prompt(
    label: str,
    user_question: str,
    box: Optional[tuple] = None,
    image_size: Optional[tuple] = None,
    inpainting_prompt: str = "",
) -> str:
    """
    VLM 프롬프트 생성.

    Args:
        label:              제거된 객체 이름
        user_question:      사용자 추가 질문
        box:                바운딩 박스 좌표 (x_min, y_min, x_max, y_max)
        image_size:         원본 이미지 크기 (width, height)
        inpainting_prompt:  인페인팅 시 사용자가 입력한 프롬프트

    Returns:
        최종 프롬프트 문자열
    """
    # 위치 정보: 좌표 대신 이미지 내 상대적 위치를 VLM이 설명하도록 유도
    location_info = ""
    if box and image_size:
        x_min, y_min, x_max, y_max = box
        w, h = image_size
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        location_info = (
            f" In the original image (size: {w}x{h} pixels), "
            f"the '{label}' was centered at pixel ({int(cx)}, {int(cy)}). "
            f"Please describe its location in natural language "
            f"(e.g., upper-left, center, bottom-right, etc.) "
            f"rather than using raw coordinates."
        )

    # 인페인팅 프롬프트 정보
    inpaint_info = ""
    if inpainting_prompt.strip():
        inpaint_info = (
            f" The user filled the removed area using the following prompt: "
            f'"{inpainting_prompt.strip()}".'
        )
    else:
        inpaint_info = " The removed area was filled automatically without a specific prompt."

    base = (
        f"You are given two images.\n"
        f"- The first image is the original full photo.\n"
        f"- The second image is the cropped region from the original where a '{label}' "
        f"was detected, segmented, and removed via inpainting."
        f"{location_info}"
        f"{inpaint_info}\n\n"
        f"Please summarize:\n"
        f"1. What object was selected and where it was located in the original image\n"
        f"2. What the removed area was replaced with\n"
        f"3. How natural or successful the result appears to be"
    )

    if user_question.strip():
        base += f"\n\nAdditional question: {user_question.strip()}"

    return base


# ──────────────────────────────────────────────
# VLM 추론
# ──────────────────────────────────────────────

def run_vlm(
    original_image: Image.Image,
    final_image: Image.Image,
    label: str,
    user_question: str = "",
    max_new_tokens: int = 512,
    box: Optional[tuple] = None,
    image_size: Optional[tuple] = None,
    inpainting_prompt: str = "",
) -> str:
    """
    Qwen2.5-VL로 두 이미지를 비교하여 요약 생성.

    Args:
        original_image:  원본 전체 이미지
        final_image:     인페인팅 후 최종 이미지
        label:           제거된 객체 이름
        user_question:   사용자 추가 질문
        max_new_tokens:  최대 생성 토큰 수

    Returns:
        요약 텍스트
    """
    model, processor = load_model()

    prompt = build_prompt(
        label=label,
        user_question=user_question,
        box=box,
        image_size=image_size,
        inpainting_prompt=inpainting_prompt,
    )

    # Qwen2.5-VL 멀티이미지 메시지 형식
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": original_image},
                {"type": "image", "image": final_image},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    # 채팅 템플릿 적용
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 이미지 + 텍스트 전처리
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE)

    print(f"[vlm] 추론 중... label='{label}'")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    # 입력 토큰 제외하고 생성된 부분만 디코딩
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    result = processor.batch_decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print("[vlm] 추론 완료.")
    return result.strip()


# ──────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────

_vlm_state = {
    "original_image":    None,   # PIL 원본 이미지
    "final_image":       None,   # PIL 인페인팅된 크롭 이미지
    "label":             "",     # 제거된 객체 이름
    "box":               None,   # 바운딩 박스 좌표
    "image_size":        None,   # 원본 이미지 크기
    "inpainting_prompt": "",     # 인페인팅 프롬프트
    "result":            "",     # VLM 요약 결과
}


def set_vlm_inputs(
    original_image: Image.Image,
    final_image: Image.Image,
    label: str,
    box: Optional[tuple] = None,
    inpainting_prompt: str = "",
):
    """Inpainting → VLM 데이터 전달 시 호출."""
    _vlm_state["original_image"]    = original_image
    _vlm_state["final_image"]       = final_image
    _vlm_state["label"]             = label
    _vlm_state["box"]               = box
    _vlm_state["image_size"]        = original_image.size if original_image else None
    _vlm_state["inpainting_prompt"] = inpainting_prompt
    _vlm_state["result"]            = ""


# ──────────────────────────────────────────────
# Gradio 콜백
# ──────────────────────────────────────────────

def gradio_run(user_question: str, max_tokens: int):
    """VLM 요약 실행."""
    if _vlm_state["original_image"] is None or _vlm_state["final_image"] is None:
        return "⚠️  Inpainting 탭에서 결과를 생성하고 전달하세요."

    label = _vlm_state["label"] or "object"

    summary = run_vlm(
        _vlm_state["original_image"],
        _vlm_state["final_image"],
        label=label,
        user_question=user_question,
        max_new_tokens=max_tokens,
        box=_vlm_state["box"],
        image_size=_vlm_state["image_size"],
        inpainting_prompt=_vlm_state["inpainting_prompt"],
    )
    _vlm_state["result"] = summary

    return summary


# ──────────────────────────────────────────────
# UI 빌드
# ──────────────────────────────────────────────

def build_ui() -> gr.Blocks:

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    * { box-sizing: border-box; }
    body, .gradio-container { background: #080b14 !important; font-family: 'Syne', sans-serif !important; }
    .pipeline-badge { display: inline-block; background: #0a1e35; border: 1px solid #1a4060; border-radius: 20px; padding: 3px 12px; font-family: 'Space Mono', monospace; font-size: 0.68rem; color: #4a9fd4; letter-spacing: 0.1em; margin: 2px; }
    .pipeline-badge.active { background: #0d2e50; border-color: #1a6fff; color: #7ecfff; box-shadow: 0 0 8px rgba(30,120,255,0.3); }
    """

    with gr.Blocks(css=css, title="VLM Summary") as demo:

        gr.HTML("""
        <div style="text-align:center; padding:24px 0 12px 0;">
            <h1 style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;color:#e0f0ff;margin:0;">
                VLM Pipeline Summary
            </h1>
            <p style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#3a6f9a;
                      text-transform:uppercase;letter-spacing:0.1em;margin:4px 0 8px 0;">
                Qwen2.5-VL · Step 4 of 4
            </p>
            <div>
                <span class="pipeline-badge">① DETECTION</span>
                <span class="pipeline-badge">② SEGMENTATION</span>
                <span class="pipeline-badge">③ INPAINTING</span>
                <span class="pipeline-badge active">④ VLM SUMMARY</span>
            </div>
        </div>
        """)

        with gr.Row():
            # 왼쪽: 이미지 비교
            with gr.Column(scale=1):
                v_original = gr.Image(label="원본 이미지", type="numpy", height=280)
                v_final    = gr.Image(label="최종 결과 이미지", type="numpy", height=280)

            # 오른쪽: 입력 + 요약 결과
            with gr.Column(scale=1):
                v_label = gr.Textbox(
                    label="제거된 객체 (자동 입력)",
                    interactive=False,
                    lines=1,
                )
                v_question = gr.Textbox(
                    label="추가 질문 (선택 — 비워두면 기본 요약)",
                    placeholder="예) Was the removal natural? What was behind the person?",
                    lines=3,
                )
                v_max_tokens = gr.Slider(
                    128, 1024, value=512, step=64,
                    label="최대 생성 토큰 수",
                )
                v_run_btn = gr.Button("🔍  요약 생성", variant="primary")
                v_summary = gr.Textbox(
                    label="VLM 요약 결과",
                    lines=12,
                    interactive=False,
                )

        v_run_btn.click(
            fn=gradio_run,
            inputs=[v_question, v_max_tokens],
            outputs=[v_summary, v_original, v_final],
        )

    return demo


# ──────────────────────────────────────────────
# 단독 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  VLM Pipeline Summary — vlm.py")
    print(f"  모델: {MODEL_ID}")
    print(f"  디바이스: {DEVICE}")
    print("=" * 60)
    load_model()
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7863, show_error=True, inbrowser=True)