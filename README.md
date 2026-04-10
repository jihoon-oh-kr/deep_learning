# 🧹 Object Removal Pipeline

> **Detection · Segmentation · Inpainting · VLM** 을 하나의 Gradio UI로 통합한 이미지 객체 제거 파이프라인

---

## 📌 개요

이미지에서 원하는 객체를 탐지(Detection)하고, 정밀하게 세그멘테이션(Segmentation)한 뒤, 해당 영역을 자연스럽게 인페인팅(Inpainting)으로 채우고, VLM이 결과를 요약하는 4단계 파이프라인입니다.

각 모듈은 독립적으로 실행할 수 있으며, `main.py`를 통해 4단계 전체를 하나의 UI에서 연결하여 실행할 수도 있습니다.

---

## 🔧 사용 모델

| 단계 | 모델 | 역할 |
|------|------|------|
| ① Detection | [Grounding DINO tiny](https://github.com/IDEA-Research/GroundingDINO) | 자유 텍스트로 객체 탐지 |
| ② Segmentation | [SAM2 hiera-tiny](https://github.com/facebookresearch/segment-anything-2) | 클릭 기반 인터랙티브 세그멘테이션 |
| ③ Inpainting | [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) | 마스크 영역 배경 복원 (파이프라인 전용) |
| ④ VLM Summary | [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 이미지 기반 자연어 질의응답 |

---

## 🚀 빠른 시작

### 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 🔬 모듈 단독 실행 (개별 데모)

각 파일을 직접 실행하면 해당 모델만 독립적으로 동작합니다.  
파이프라인 연결 없이 각 파운데이션 모델의 동작을 개별적으로 확인할 수 있습니다.

> **참고:** Inpainting(`inpainting.py`)은 Segmentation이 생성한 마스크를 필수 입력으로 사용하기 때문에 단독 실행을 지원하지 않습니다. `main.py`의 통합 파이프라인에서만 동작합니다.

---

### ① detection.py

```bash
python detection.py
# 접속 주소: http://localhost:7860
```

| 항목 | 내용 |
|------|------|
| **입력** | 이미지 파일, 탐지할 객체 텍스트 (`. `으로 구분, 예: `person . car . dog`) |
| **출력** | 바운딩 박스가 그려진 결과 이미지, 탐지된 객체 목록 및 신뢰도, 선택한 객체의 크롭 이미지 |

**사용 방법**
1. 이미지를 업로드합니다.
2. 탐지할 객체를 텍스트로 입력합니다.
3. **탐지 실행** 버튼을 클릭합니다.
4. 드롭다운에서 확인할 객체를 선택하면 크롭 이미지와 바운딩 박스 정보가 표시됩니다.

---

### ② segmentation.py

```bash
python segmentation.py
# 접속 주소: http://localhost:7861
```

| 항목 | 내용 |
|------|------|
| **입력** | 이미지 파일, 클릭으로 지정하는 포인트 (1개 이상) |
| **출력** | 마스크 오버레이 이미지, 객체가 제거된 배경 이미지, 최종 마스크 |

**사용 방법**
1. 이미지를 업로드합니다.
2. 세그멘테이션할 객체 위를 클릭합니다. (클릭할 때마다 마스크가 즉시 업데이트됩니다)
3. 마스크가 마음에 들지 않으면 **마지막 포인트 취소** 또는 **전체 초기화**를 누릅니다.
4. 마스크가 만족스러우면 **확인 (마스크 확정)** 버튼을 클릭합니다.

---

### ③ vlm.py

```bash
python vlm.py
# 접속 주소: http://localhost:7863
```

| 항목 | 내용 |
|------|------|
| **입력** | 이미지 파일 1장, 질문 텍스트 (선택 — 비워두면 이미지 전체 설명 요청) |
| **출력** | 이미지에 대한 VLM 자연어 답변 |

**사용 방법**
1. 이미지를 업로드합니다.
2. 질문을 입력합니다. (예: `What is in this image?` / 비워두면 자동으로 이미지 설명)
3. **실행** 버튼을 클릭합니다.

---

## 🔗 통합 파이프라인 실행 (main.py)

4개의 모듈을 하나의 UI로 연결한 전체 파이프라인입니다.  
실행 시 4개의 모델이 순차적으로 로드되고, 브라우저가 자동으로 열립니다.

```bash
python main.py
# 접속 주소: http://localhost:7860
```

### 파이프라인 흐름

**Step 1 — Detection (탭 ①)**
1. 이미지를 업로드하고 탐지할 객체를 텍스트로 입력합니다.
2. **탐지 실행** 버튼을 클릭합니다.
3. 탐지된 객체 목록에서 제거할 객체를 선택합니다.
4. **Step 2: Segmentation으로 전달** 버튼을 클릭합니다.

**Step 2 — Segmentation (탭 ②)**
1. 전달된 크롭 이미지 위를 클릭하여 세그멘테이션 포인트를 지정합니다.
2. 마스크가 만족스러우면 **확인 (마스크 확정)** 을 클릭합니다.
3. **Step 3: Inpainting으로 전달** 버튼을 클릭합니다.

**Step 3 — Inpainting (탭 ③)**
1. 채울 배경을 설명하는 프롬프트를 입력합니다.
2. **인페인팅 실행** 버튼을 클릭합니다.
3. 결과가 만족스러우면 **Step 4: VLM Summary로 전달** 버튼을 클릭합니다.

**Step 4 — VLM Summary (탭 ④)**
1. 원본 이미지와 최종 결과 이미지가 자동으로 표시됩니다.
2. 추가 질문을 입력하거나 비워두면 기본 요약이 생성됩니다.
3. **요약 생성** 버튼을 클릭합니다.

---

## 📁 프로젝트 구조

```
.
├── main.py            # 통합 파이프라인 진입점 & Gradio UI 빌더
├── detection.py       # Grounding DINO 탐지 모듈 (단독 실행 가능)
├── segmentation.py    # SAM2 세그멘테이션 모듈 (단독 실행 가능)
├── inpainting.py      # Stable Diffusion 인페인팅 모듈 (파이프라인 전용)
├── vlm.py             # Qwen2.5-VL VLM 모듈 (단독 실행 가능)
├── utils.py           # 공유 상태(PipelineState) 및 유틸리티
└── requirements.txt   # 패키지 목록
```

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
