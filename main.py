"""
main.py
-------
Detection → Segmentation → Inpainting → VLM 통합 파이프라인.
실행: python main.py
"""

from __future__ import annotations

import sys
import os
import numpy as np
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detection
import segmentation
import inpainting
import vlm
from utils import PipelineState, apply_mask_to_image

pipeline = PipelineState()


# ──────────────────────────────────────────────
# 탭 간 전달 함수
# ──────────────────────────────────────────────

def send_to_segmentation():
    """
    Detection 탭 → Segmentation 탭.
    detection._state.cropped_image를 segmentation으로 전달.
    """
    det = detection._state

    if det.cropped_image is None:
        return None, "⚠️  Detection 탭에서 객체를 선택한 뒤 눌러주세요."

    # pipeline 공유 상태 업데이트
    pipeline.original_image     = det.original_image
    pipeline.detection_results  = det.detection_results
    pipeline.selected_detection = det.selected_detection
    pipeline.cropped_image      = det.cropped_image

    # segmentation 모듈에 이미지 주입
    segmentation.set_seg_image(det.cropped_image)

    label  = det.selected_detection.label if det.selected_detection else "unknown"
    status = (
        f"'{label}' 이미지가 세그멘테이션 탭으로 전달됐습니다.\n"
        f"② Segmentation 탭으로 이동해서 클릭하세요."
    )

    # 세그멘테이션 탭의 이미지 컴포넌트에 표시할 오버레이 반환
    overlay = segmentation.render_overlay(det.cropped_image, [], None)
    return overlay, status


def send_to_inpainting():
    """Segmentation 탭 → Inpainting 탭 (준비 중)."""
    seg = segmentation._seg_state

    if seg["final_mask"] is None:
        return None, None, "⚠️  Segmentation 탭에서 마스크를 확정하세요."

    pipeline.segmentation_mask = seg["final_mask"]
    pipeline.cropped_image     = seg["image"]
    inpainting.set_inp_inputs(
        seg["image"],
        seg["final_mask"],
        original_image=pipeline.original_image,
        box=pipeline.selected_detection.box if pipeline.selected_detection else None,
    )

    bg       = apply_mask_to_image(seg["image"], seg["final_mask"])
    pipeline.background_only = bg

    mask_vis = Image.fromarray((seg["final_mask"] * 255).astype(np.uint8)).convert("RGB")
    status   = "이미지/마스크가 Inpainting 탭으로 전달됐습니다. ③ 탭으로 이동하세요."
    return np.array(bg), np.array(mask_vis), status


# ──────────────────────────────────────────────
# 메인 UI
# ──────────────────────────────────────────────

def build_main_ui():

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
    * { box-sizing: border-box; }
    body, .gradio-container { background: #060910 !important; font-family: 'Syne', sans-serif !important; }
    .tab-nav button { font-family: 'Space Mono', monospace !important; font-size: 0.78rem !important;
                      letter-spacing: 0.08em !important; text-transform: uppercase !important; }
    .hint-box { font-family: 'Space Mono', monospace; font-size: 0.72rem; color: #5a9abf;
                background: #0a1520; border: 1px solid #1a3a5c; border-radius: 8px;
                padding: 10px 14px; margin-bottom: 10px; line-height: 1.9; }
    """

    with gr.Blocks(css=css, title="Object Removal Pipeline") as demo:

        gr.HTML("""
        <div style="text-align:center; padding:28px 0 8px 0;">
            <h1 style="font-family:'Syne',sans-serif;font-weight:800;font-size:2.4rem;
                       color:#e0f0ff;letter-spacing:-0.02em;margin:0;
                       text-shadow:0 0 40px rgba(100,180,255,0.3);">
                Object Removal Pipeline
            </h1>
            <p style="font-family:'Space Mono',monospace;font-size:0.72rem;
                      color:#3a6f9a;letter-spacing:0.12em;text-transform:uppercase;margin:6px 0 0 0;">
                Detection · Segmentation · Inpainting · VLM Summary
            </p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Detection ──────────────────────────────────
            with gr.Tab("① Detection"):

                gr.HTML('<p style="font-family:\'Space Mono\',monospace;font-size:0.7rem;color:#3a6f9a;letter-spacing:0.1em;text-transform:uppercase;padding:8px 0 4px 0;">Step 1 — Open-Vocabulary Object Detection (Grounding DINO)</p>')

                with gr.Row():
                    with gr.Column(scale=1):
                        d_image   = gr.Image(label="이미지 업로드", type="numpy", height=320)
                        d_prompt  = gr.Textbox(
                            label="탐지할 객체 ('. '으로 구분 — 자유 입력)",
                            value="", lines=2,
                            placeholder="예)  person . car . dog . cat",
                        )
                        d_btn     = gr.Button("🔍  탐지 실행", variant="primary")

                    with gr.Column(scale=1):
                        d_result  = gr.Image(label="탐지 결과", type="numpy", height=320)
                        d_summary = gr.Textbox(label="탐지 요약", lines=5, interactive=False)

                gr.HTML('<hr style="border-color:#1a2a3a;margin:16px 0;">')

                with gr.Row():
                    with gr.Column(scale=1):
                        d_selector = gr.Dropdown(label="탐지된 객체 선택", choices=[], interactive=True)
                        d_boxinfo  = gr.Textbox(label="바운딩 박스 정보", lines=7, interactive=False)

                    with gr.Column(scale=1):
                        d_cropped  = gr.Image(label="선택된 객체 크롭", type="numpy", height=240)
                        d_transfer = gr.Button("▶  Step 2: Segmentation으로 전달", variant="primary")
                        d_t_status = gr.Textbox(label="전달 상태", lines=2, interactive=False)

                # Detection 이벤트
                d_btn.click(
                    fn=detection.gradio_detect,
                    inputs=[d_image, d_prompt],
                    outputs=[d_result, d_summary, d_selector, d_boxinfo],
                )
                d_selector.change(
                    fn=detection.gradio_select_box,
                    inputs=[d_selector],
                    outputs=[d_cropped, d_boxinfo],
                )

            # ── Tab 2: Segmentation ───────────────────────────────
            with gr.Tab("② Segmentation"):

                gr.HTML('<p style="font-family:\'Space Mono\',monospace;font-size:0.7rem;color:#3a6f9a;letter-spacing:0.1em;text-transform:uppercase;padding:8px 0 4px 0;">Step 2 — Interactive Segmentation (SAM2)</p>')

                gr.HTML("""
                <div class="hint-box">
                    🖱️ <b>좌클릭</b> — 포인트 추가 → 마스크 즉시 업데이트 &nbsp;|&nbsp;
                    ↩️ <b>마지막 포인트 취소</b> — 되돌리기 &nbsp;|&nbsp;
                    ✅ <b>확인</b> — 마스크 확정
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        s_image   = gr.Image(
                            label="클릭해서 세그멘테이션 영역 선택 (Detection에서 자동 로드됨)",
                            type="numpy", interactive=True,
                        )
                        with gr.Row():
                            s_confirm = gr.Button("✅  확인 (마스크 확정)", variant="primary")
                            s_undo    = gr.Button("↩️  마지막 포인트 취소", variant="secondary")
                            s_reset   = gr.Button("🔄  초기화", variant="secondary")
                        s_status  = gr.Textbox(label="상태", lines=3, interactive=False)

                    with gr.Column(scale=1):
                        s_bg      = gr.Image(label="배경만 남긴 이미지", type="numpy", height=200)
                        s_mask    = gr.Image(label="최종 마스크", type="numpy", height=200)
                        s_transfer = gr.Button("▶  Step 3: Inpainting으로 전달", variant="primary")
                        s_t_status = gr.Textbox(label="전달 상태", lines=2, interactive=False)

                # Segmentation 이벤트
                s_image.select(fn=segmentation.gradio_click,    outputs=[s_image, s_status])
                s_undo.click(   fn=segmentation.gradio_undo,     outputs=[s_image, s_status])
                s_confirm.click(fn=segmentation.gradio_confirm,  outputs=[s_bg, s_mask, s_status])
                s_reset.click(  fn=segmentation.gradio_reset,    outputs=[s_image, s_status])
                s_transfer.click(fn=send_to_inpainting, outputs=[s_bg, s_mask, s_t_status])

            # ── Tab 3: Inpainting ─────────────────────────────────
            with gr.Tab("③ Inpainting"):

                gr.HTML('<p style="font-family:\'Space Mono\',monospace;font-size:0.7rem;color:#3a6f9a;letter-spacing:0.1em;text-transform:uppercase;padding:8px 0 4px 0;">Step 3 — Background Inpainting (Stable Diffusion 1.5)</p>')

                with gr.Row():
                    with gr.Column(scale=1):
                        i_bg      = gr.Image(label="객체 제거된 배경 (입력)", type="numpy", height=200)
                        i_mask    = gr.Image(label="마스크", type="numpy", height=200)
                        i_prompt  = gr.Textbox(
                            label="채울 배경 설명 (구체적일수록 좋음)",
                            value="empty background, no people, natural scene, realistic",
                            placeholder="예) grass field, concrete floor, wooden wall, empty street",
                            lines=2,
                        )
                        i_neg     = gr.Textbox(
                            label="네거티브 프롬프트",
                            value="person, human, people, face, body, blurry, bad quality, distorted, watermark",
                            lines=2,
                        )
                        with gr.Row():
                            i_steps = gr.Slider(10, 50, value=30, step=5, label="스텝 수")
                            i_guide = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="가이던스")
                        i_seed    = gr.Slider(-1, 9999, value=-1, step=1, label="시드 (-1=랜덤)")
                        i_run_btn = gr.Button("🎨  인페인팅 실행", variant="primary")

                    with gr.Column(scale=1):
                        i_result   = gr.Image(label="인페인팅 결과 (크롭)", type="numpy", height=280)
                        i_full     = gr.Image(label="전체 이미지 (원본에 합성)", type="numpy", height=280)
                        i_status   = gr.Textbox(label="상태", lines=3, interactive=False)
                        i_transfer = gr.Button("▶  Step 4: VLM Summary로 전달", variant="primary")
                        i_t_status = gr.Textbox(label="전달 상태", lines=2, interactive=False)

                i_run_btn.click(
                    fn=inpainting.gradio_run,
                    inputs=[i_prompt, i_neg, i_steps, i_guide, i_seed],
                    outputs=[i_result, i_full, i_status],
                )

            # ── Tab 4: VLM Summary ────────────────────────────────
            with gr.Tab("④ VLM Summary"):

                gr.HTML('<p style="font-family:Space Mono,monospace;font-size:0.7rem;color:#3a6f9a;letter-spacing:0.1em;text-transform:uppercase;padding:8px 0 4px 0;">Step 4 — Pipeline Summary (Qwen2.5-VL)</p>')

                with gr.Row():
                    with gr.Column(scale=1):
                        v_original = gr.Image(label="원본 이미지", type="numpy", height=280)
                        v_final    = gr.Image(label="최종 결과 이미지", type="numpy", height=280)
                    with gr.Column(scale=1):
                        v_label    = gr.Textbox(label="제거된 객체 (자동)", interactive=False, lines=1)
                        v_question = gr.Textbox(
                            label="추가 질문 (선택 — 비워두면 기본 요약)",
                            placeholder="예) Was the removal natural?",
                            lines=3,
                        )
                        v_tokens   = gr.Slider(128, 1024, value=512, step=64, label="최대 생성 토큰 수")
                        v_run_btn  = gr.Button("🔍  요약 생성", variant="primary")
                        v_summary  = gr.Textbox(label="VLM 요약 결과", lines=10, interactive=False)

        # ── Detection → Segmentation 전달 (탭 밖에서 연결) ──
        d_transfer.click(
            fn=send_to_segmentation,
            outputs=[s_image, d_t_status],
        )

        # ── Segmentation → Inpainting 전달 시 i_bg, i_mask 업데이트 ──
        def load_inpainting_preview():
            if inpainting._inp_state["image"] is None:
                return None, None
            bg = apply_mask_to_image(inpainting._inp_state["image"], inpainting._inp_state["mask"])
            mask_vis = Image.fromarray((inpainting._inp_state["mask"] * 255).astype(np.uint8)).convert("RGB")
            return np.array(bg), np.array(mask_vis)

        s_transfer.click(
            fn=load_inpainting_preview,
            outputs=[i_bg, i_mask],
        )

        # ── Inpainting → VLM 전달 ──
        def send_to_vlm():
            label      = pipeline.selected_detection.label if pipeline.selected_detection else "object"
            box        = pipeline.selected_detection.box   if pipeline.selected_detection else None
            orig       = pipeline.original_image
            crop_result = inpainting._inp_state["result"]   # 크롭 인페인팅 결과 (VLM 입력용)
            full_result = None
            inp_prompt  = inpainting._inp_state.get("last_prompt", "")

            if orig is None or crop_result is None:
                return None, None, "", "⚠️  Inpainting 결과가 없습니다."

            if isinstance(crop_result, np.ndarray):
                crop_result = Image.fromarray(crop_result)

            # 전체 합성 이미지 생성 (원본에 인페인팅 결과 붙이기)
            if box is not None:
                full_result = inpainting.compose_full_image(crop_result, orig, box)
            else:
                full_result = orig

            # VLM에는 크롭 결과 전달 (두 번째 이미지)
            vlm.set_vlm_inputs(
                orig, crop_result, label,
                box=box,
                inpainting_prompt=inp_prompt,
            )
            status = (
                f"VLM 탭으로 전달 완료!\n"
                f"객체: '{label}'  박스: {box}\n"
                f"인페인팅 프롬프트: '{inp_prompt if inp_prompt else '(없음)'}'"
            )
            # v_original: 원본, v_final: 전체 합성 이미지 (UI 표시용)
            return np.array(orig), np.array(full_result), label, status

        i_transfer.click(fn=send_to_vlm, outputs=[v_original, v_final, v_label, i_t_status])

        v_run_btn.click(
            fn=vlm.gradio_run,
            inputs=[v_question, v_tokens],
            outputs=[v_summary],
        )

    return demo


# ──────────────────────────────────────────────
# 엔트리포인트
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Object Removal Pipeline — main.py")
    print("=" * 60)

    print("\n[1/3] Detection 모델 로드...")
    detection.load_model()

    print("\n[2/3] Segmentation 모델 로드...")
    segmentation.load_model()

    print("\n[3/4] Inpainting 모델 로드...")
    inpainting.load_model()

    print("\n[4/4] VLM 모델 로드...")
    vlm.load_model()

    print("\n모든 모델 로드 완료.\n")

    app = build_main_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        inbrowser=True,
    )