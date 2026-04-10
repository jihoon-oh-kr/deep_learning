# 🧹 Object Removal Pipeline

> **Detection · Segmentation · Inpainting · VLM** 을 하나의 Gradio UI로 통합한 이미지 객체 제거 파이프라인

---

## 📌 개요

이미지에서 원하는 객체를 탐지(Detection)하고, 정밀하게 세그멘테이션(Segmentation)한 뒤, 해당 영역을 자연스럽게 인페인팅(Inpainting)으로 채우고, VLM이 결과를 요약하는 4단계 파이프라인입니다.

---

## 🔧 사용 모델

| 단계 | 모델 | 역할 |
|------|------|------|
| ① Detection | [Grounding DINO tiny](https://github.com/IDEA-Research/GroundingDINO) | 자유 텍스트로 객체 탐지 |
| ② Segmentation | [SAM2 hiera-tiny](https://github.com/facebookresearch/segment-anything-2) | 클릭 기반 인터랙티브 세그멘테이션 |
| ③ Inpainting | [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | 마스크 영역 배경 복원 |
| ④ VLM Summary | [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 파이프라인 결과 자연어 요약 |

---

## 🚀 빠른 시작

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 실행

```bash
python main.py
```

실행 시 4개의 모델이 순차적으로 로드되고, 브라우저가 자동으로 열립니다.  
기본 주소: `http://localhost:7860`

---

## 🖼️ 사용 방법

### Step 1 — Detection (탭 ①)
1. 이미지를 업로드합니다.
2. 탐지할 객체를 텍스트로 입력합니다. (`. ` 으로 구분, 예: `person . car . dog`)
3. **탐지 실행** 버튼을 클릭합니다.
4. 탐지된 객체 목록에서 제거할 객체를 선택합니다.
5. **Step 2: Segmentation으로 전달** 버튼을 클릭합니다.

### Step 2 — Segmentation (탭 ②)
1. 전달된 크롭 이미지 위를 클릭하여 세그멘테이션 포인트를 지정합니다.
2. 마스크가 만족스러우면 **확인 (마스크 확정)** 을 클릭합니다.
3. **Step 3: Inpainting으로 전달** 버튼을 클릭합니다.

### Step 3 — Inpainting (탭 ③)
1. 채울 배경을 설명하는 프롬프트를 입력합니다. (구체적일수록 결과가 좋습니다)
2. 스텝 수, 가이던스, 시드 등 파라미터를 조정합니다.
3. **인페인팅 실행** 버튼을 클릭합니다.
4. 결과가 만족스러우면 **Step 4: VLM Summary로 전달** 버튼을 클릭합니다.

### Step 4 — VLM Summary (탭 ④)
1. 원본 이미지와 최종 결과 이미지가 자동으로 표시됩니다.
2. 추가 질문을 입력하거나 비워두면 기본 요약이 생성됩니다.
3. **요약 생성** 버튼을 클릭합니다.

---

## 📁 프로젝트 구조

```
.
├── main.py            # 메인 진입점 & Gradio UI 빌더
├── detection.py       # Grounding DINO 탐지 모듈
├── segmentation.py    # SAM2 세그멘테이션 모듈
├── inpainting.py      # Stable Diffusion 인페인팅 모듈
├── vlm.py             # Qwen2.5-VL 요약 모듈
├── utils.py           # 공유 상태(PipelineState) 및 유틸리티
└── requirements.txt   # 패키지 목록
```

---

## ⚙️ 주요 파라미터 (Inpainting)

| 파라미터 | 범위 | 기본값 | 설명 |
|----------|------|--------|------|
| 스텝 수 | 10 ~ 50 | 30 | 디퓨전 스텝 수 (높을수록 품질↑, 속도↓) |
| 가이던스 | 1.0 ~ 15.0 | 7.5 | 프롬프트 충실도 (높을수록 프롬프트 강하게 반영) |
| 시드 | -1 ~ 9999 | -1 | -1이면 랜덤 시드 |

---

## 📋 요구 사항

- Python 3.9+
- CUDA 지원 GPU 권장 (CPU에서도 동작하나 매우 느림)
- 최소 VRAM: 6GB 이상 권장

---

## 🛑 종료

```
Ctrl+C
```
