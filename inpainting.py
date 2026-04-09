"""
inpainting.py
-------------
Stable Diffusion 1.5 Inpainting 기반 배경 복원 모듈.

입력:
  - cropped_image:      바운딩 박스 크롭 이미지 (PIL)
  - segmentation_mask:  SAM2 이진 마스크 (1=객체/채울 영역, 0=배경)
  - prompt:             사용자 텍스트 프롬프트 (선택)

출력:
  - 객체가 제거되고 배경이 자연스럽게 채워진 이미지

모델: runwayml/stable-diffusion-inpainting
  - VRAM ~4GB, RAM ~6GB
  - 512x512 입력 권장
"""

from __future__ import annotations

import warnings, logging, os
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["TQDM_DISABLE"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


import os
import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image, ImageFilter
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import PipelineState, mask_to_pil

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

MODEL_ID   = "runwayml/stable-diffusion-inpainting"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
INPAINT_SIZE = 512   # SD 1.5 권장 입력 크기

# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────

_pipe = None

def load_model():
    global _pipe
    if _pipe is not None:
        return _pipe

    import transformers
    import diffusers as _diffusers

    # torch.load 보안 체크 무력화 (torch < 2.6 환경에서 필요)
    os.environ["TRANSFORMERS_VERIFY_SCHEDULED_UPDATES"] = "0"
    def _skip_check(*args, **kwargs):
        return True
    try:
        transformers.utils.import_utils.check_torch_load_is_safe = _skip_check
        transformers.modeling_utils.check_torch_load_is_safe = _skip_check
    except AttributeError:
        pass

    # diffusers 진행바 및 경고 완전 억제
    _diffusers.logging.set_verbosity_error()
    os.environ["DIFFUSERS_VERBOSITY"] = "error"

    from diffusers import StableDiffusionInpaintPipeline
    import logging as _logging
    # "An error occurred while trying to fetch" 메시지 억제
    _logging.getLogger("diffusers.pipelines.pipeline_utils").setLevel(_logging.CRITICAL)
    _logging.getLogger("diffusers").setLevel(_logging.CRITICAL)

    _pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE)

    # 메모리 최적화
    if DEVICE == "cuda":
        _pipe.enable_attention_slicing()

    return _pipe


# ──────────────────────────────────────────────
# 전처리 / 후처리
# ──────────────────────────────────────────────

def preprocess(
    image: Image.Image,
    mask: np.ndarray,
) -> tuple[Image.Image, Image.Image, int, int]:
    """
    이미지와 마스크를 512x512로 리사이즈.
    원본 크기를 반환해서 나중에 복원할 수 있게 함.
    """
    orig_w, orig_h = image.size

    image_resized = image.resize((INPAINT_SIZE, INPAINT_SIZE), Image.LANCZOS)

    # 마스크: 1=채울 영역(객체), 0=유지할 영역(배경)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    # 마스크 경계를 약간 dilate해서 경계선 흔적 방지
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(7))
    mask_resized = mask_pil.resize((INPAINT_SIZE, INPAINT_SIZE), Image.NEAREST)

    return image_resized, mask_resized, orig_w, orig_h


def postprocess(
    inpainted: Image.Image,
    original: Image.Image,
    mask: np.ndarray,
    orig_w: int,
    orig_h: int,
) -> Image.Image:
    """
    인페인팅 결과를 원본 크기로 복원하고,
    마스크 바깥(배경) 영역은 원본 픽셀로 덮어씌워 배경 왜곡 방지.
    """
    # 원본 크기로 리사이즈
    inpainted_resized = inpainted.resize((orig_w, orig_h), Image.LANCZOS)

    # 마스크 바깥 영역 = 원본 픽셀로 복원
    orig_arr     = np.array(original)
    inpaint_arr  = np.array(inpainted_resized)
    mask_bool    = (mask > 0).astype(bool)    # True = 채운 영역

    result = orig_arr.copy()
    result[mask_bool] = inpaint_arr[mask_bool]

    return Image.fromarray(result)


# ──────────────────────────────────────────────
# 인페인팅 핵심 함수
# ──────────────────────────────────────────────

def run_inpainting(
    image: Image.Image,
    mask: np.ndarray,
    prompt: str = "",
    negative_prompt: str = "person, human, people, face, body, blurry, bad quality, distorted, watermark",
    num_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = -1,
) -> Image.Image:
    """
    SD 1.5 Inpainting으로 마스크 영역을 자연스럽게 채움.

    Args:
        image:           원본 크롭 PIL 이미지
        mask:            이진 마스크 numpy (1=채울 영역)
        prompt:          채울 내용 설명 (빈 문자열이면 배경 문맥 자동 추론)
        negative_prompt: 생성에서 제외할 요소
        num_steps:       diffusion 스텝 수 (많을수록 품질↑, 속도↓)
        guidance_scale:  프롬프트 가이던스 강도
        seed:            랜덤 시드 (-1이면 랜덤)

    Returns:
        인페인팅된 PIL 이미지 (원본 크기)
    """
    pipe = load_model()

    image_input, mask_input, orig_w, orig_h = preprocess(image, mask)

    # 시드 설정
    generator = None
    if seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)


    result = pipe(
        prompt=prompt.strip(),
        negative_prompt=negative_prompt,
        image=image_input,
        mask_image=mask_input,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    # 후처리: 원본 크기 복원 + 배경 보존
    final = postprocess(result, image, mask, orig_w, orig_h)
    return final


# ──────────────────────────────────────────────
# 전역 상태
# ──────────────────────────────────────────────

_inp_state = {
    "image":          None,
    "mask":           None,
    "result":         None,
    "original_image": None,
    "box":            None,
    "last_prompt":    "",    # 마지막으로 사용한 인페인팅 프롬프트
}

def set_inp_inputs(image: Image.Image, mask: np.ndarray,
                   original_image: Optional[Image.Image] = None,
                   box: Optional[tuple] = None):
    """Segmentation → Inpainting 이미지/마스크 전달 시 호출."""
    _inp_state["image"]          = image
    _inp_state["mask"]           = mask
    _inp_state["result"]         = None
    _inp_state["original_image"] = original_image
    _inp_state["box"]            = box


def compose_full_image(
    inpainted_crop: Image.Image,
    original: Image.Image,
    box: tuple,
) -> Image.Image:
    """
    인페인팅된 크롭 이미지를 원본 전체 이미지의 바운딩 박스 위치에 붙여서 반환.

    Args:
        inpainted_crop: 인페인팅 결과 PIL (크롭 크기)
        original:       원본 전체 이미지 PIL
        box:            (x_min, y_min, x_max, y_max) 바운딩 박스 픽셀 좌표

    Returns:
        전체 이미지에 인페인팅 결과가 합성된 PIL 이미지
    """
    x_min, y_min, x_max, y_max = box
    box_w = x_max - x_min
    box_h = y_max - y_min

    # 크롭 결과를 바운딩 박스 크기로 리사이즈
    patch = inpainted_crop.resize((box_w, box_h), Image.LANCZOS)

    # 원본 이미지 복사 후 해당 위치에 붙이기
    full = original.copy()
    full.paste(patch, (x_min, y_min))
    return full


# ──────────────────────────────────────────────
# Gradio 콜백
# ──────────────────────────────────────────────

def gradio_run(prompt: str, negative_prompt: str, num_steps: int, guidance_scale: float, seed: int):
    """인페인팅 실행."""
    if _inp_state["image"] is None or _inp_state["mask"] is None:
        return None, None, None, "⚠️  Segmentation 탭에서 마스크를 확정하고 전달하세요."

    from utils import apply_mask_to_image
    bg_preview = apply_mask_to_image(_inp_state["image"], _inp_state["mask"])

    result = run_inpainting(
        _inp_state["image"],
        _inp_state["mask"],
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    _inp_state["result"]      = result
    _inp_state["last_prompt"] = prompt

    # 전체 이미지 합성
    full_result = None
    if _inp_state["original_image"] is not None and _inp_state["box"] is not None:
        full_result = compose_full_image(
            result,
            _inp_state["original_image"],
            _inp_state["box"],
        )

    status = (
        f"인페인팅 완료!\n"
        f"프롬프트: '{prompt if prompt.strip() else '(자동)'}'\n"
        f"스텝: {num_steps}  가이던스: {guidance_scale}"
    )
    return (
        np.array(result),
        np.array(full_result) if full_result is not None else None,
        status,
    )


def gradio_receive(image_np, mask_np):
    """Segmentation에서 이미지/마스크를 받아 미리보기 표시."""
    if image_np is None or mask_np is None:
        return None, None, "이미지와 마스크가 없습니다."

    pil   = Image.fromarray(image_np).convert("RGB")
    mask  = (np.array(Image.fromarray(mask_np).convert("L")) > 127).astype(np.uint8)
    set_inp_inputs(pil, mask)

    from utils import apply_mask_to_image
    bg = apply_mask_to_image(pil, mask)
    mask_vis = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")

    return np.array(bg), np.array(mask_vis), "이미지/마스크 로드 완료. '인페인팅 실행' 버튼을 누르세요."


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

    with gr.Blocks(css=css, title="Inpainting Module") as demo:

        gr.HTML("""
        <div style="text-align:center; padding:24px 0 12px 0;">
            <h1 style="font-family:'Syne',sans-serif;font-weight:800;font-size:2rem;color:#e0f0ff;margin:0;">
                Background Inpainting
            </h1>
            <p style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#3a6f9a;
                      text-transform:uppercase;letter-spacing:0.1em;margin:4px 0 8px 0;">
                Stable Diffusion 1.5 · Step 3 of 4
            </p>
            <div>
                <span class="pipeline-badge">① DETECTION</span>
                <span class="pipeline-badge">② SEGMENTATION</span>
                <span class="pipeline-badge active">③ INPAINTING</span>
                <span class="pipeline-badge">④ VLM SUMMARY</span>
            </div>
        </div>
        """)

        with gr.Row():
            # 왼쪽: 입력 및 파라미터
            with gr.Column(scale=1):
                gr.HTML('<p style="font-family:\'Space Mono\',monospace;font-size:0.7rem;color:#3a6f9a;text-transform:uppercase;letter-spacing:0.1em;">입력 미리보기</p>')
                inp_bg   = gr.Image(label="객체 제거된 배경 (인페인팅 입력)", type="numpy", height=220)
                inp_mask = gr.Image(label="마스크 (흰색=채울 영역)", type="numpy", height=220)

                gr.HTML('<p style="font-family:\'Space Mono\',monospace;font-size:0.7rem;color:#3a6f9a;text-transform:uppercase;letter-spacing:0.1em;margin-top:12px;">프롬프트</p>')
                prompt = gr.Textbox(
                    label="채울 내용 설명 (비워두면 자동)",
                    value="",
                    placeholder="예) grass, concrete floor, wooden wall ...",
                    lines=2,
                )
                neg_prompt = gr.Textbox(
                    label="네거티브 프롬프트",
                    value="blurry, bad quality, distorted, person, watermark",
                    lines=2,
                )
                with gr.Row():
                    num_steps = gr.Slider(10, 50, value=30, step=5, label="스텝 수")
                    guidance  = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="가이던스")
                seed = gr.Slider(-1, 9999, value=-1, step=1, label="시드 (-1=랜덤)")
                run_btn = gr.Button("🎨  인페인팅 실행", variant="primary")

            # 오른쪽: 결과
            with gr.Column(scale=1):
                gr.HTML('<p style="font-family:\'Space Mono\',monospace;font-size:0.7rem;color:#3a6f9a;text-transform:uppercase;letter-spacing:0.1em;">결과</p>')
                inp_result = gr.Image(label="인페인팅 결과", type="numpy", height=460)
                inp_status = gr.Textbox(label="상태", lines=3, interactive=False)

        run_btn.click(
            fn=gradio_run,
            inputs=[prompt, neg_prompt, num_steps, guidance, seed],
            outputs=[inp_bg, inp_result, inp_status],
        )

    return demo


# ──────────────────────────────────────────────
# 단독 실행
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Background Inpainting — inpainting.py")
    print(f"  모델: {MODEL_ID}")
    print(f"  디바이스: {DEVICE}")
    print("=" * 60)
    load_model()
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7862, show_error=True, inbrowser=True)