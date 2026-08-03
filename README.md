# 🎬 OTT Spoiler Detector AI
> **사용자 경험(UX) 보호를 위한 High-Recall 스포일러 탐지 앙상블 시스템**

[![Python](https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📌 Project Overview
본 프로젝트는 영화 및 OTT 플랫폼 시청자의 몰입을 방해하는 치명적인 '스포일러'를 사전에 완벽하게 차단하여 사용자 경험을 보호하는 데 중점을 둔 AI 판독 시스템입니다. 

* **개발 기간**: 2026.05 ~ 2026.06 (개인 프로젝트)
* **핵심 기능**: 리뷰 텍스트 실시간 스포일러 위험도 분석, 대량 리뷰 일괄 필터링 대시보드, 3대장 앙상블 투표(Max Voting) 기반의 과잉 방어망 구축

---

## 📺 Demo Video & UI
> 이미지를 클릭하면 데모 영상(또는 배포 사이트)으로 이동합니다.

| 단일 리뷰 정밀 분석 (XAI) | 실시간 대량 필터링 대시보드 |
| :---: | :---: |
| [![단일분석](https://img.shields.io/badge/UI-Single_Analysis-blue?style=for-the-badge)](링크주소) | [![대량분석](https://img.shields.io/badge/UI-Batch_Dashboard-green?style=for-the-badge)](링크주소) |

* **단일 분석**: 3개의 AI 모델이 각각 분석한 위험도를 실시간 게이지 바(Progress Bar)로 시각화
* **대량 분석**: CSV 파일 업로드 시 일괄 판독 후 파이 차트(Plotly)로 직관적인 통계 제공

---

## 🏗 System Architecture
데이터 수집부터 서비스 배포까지의 전체 파이프라인을 구축했습니다.

* **Data Pipeline**: 15만 건 원본 정제 ➔ Gemma-7B 프롬프트 엔지니어링 5,000건 라벨링 ➔ 의사 레이블링(Pseudo-labeling) 기반 10만 건 데이터 확장
* **AI Architecture**: 단어 키워드(ML) + 패턴(1D-CNN) + 문맥(KcELECTRA)을 결합한 하이브리드 앙상블 아키텍처
* **Service Layer**: Streamlit 기반의 인터랙티브 웹 대시보드 배포

---

## 📊 AI Model Pipeline
서로 다른 강점을 가진 3개의 모델을 결합하여 탐지의 사각지대를 없앴습니다.

1. **TF-IDF + Logistic Regression**: '결말', '범인' 등 명시적인 스포일러 키워드 초고속 매칭 (25,000 피처)
2. **1D-CNN**: 텍스트의 구조적 패턴과 은유적 스포일러의 지역적 특징(Local Feature) 추출
3. **KcELECTRA**: 'ㅠㅠ', 'ㅋㅋ' 등 감정 기호와 앞뒤 문맥을 깊이 있게 이해하는 대형 언어 모델(Transformer)

---

## 🛠 Tech Stack
* **Language**: Python
* **AI/ML**: PyTorch, Scikit-learn, Transformers (Hugging Face)
* **Data Engineering**: Pandas, NumPy
* **LLM API**: OpenAI API (LM Studio 연동, Gemma-7B)
* **Frontend/Visualization**: Streamlit, Plotly

---

## 🌟 Key Features & Problem Solving
단순한 모델 학습을 넘어, **데이터 엔지니어링과 비즈니스 로직 최적화**를 통한 문제 해결 과정을 담았습니다.

### 1️⃣ 재현율(Recall) 극대화를 위한 앙상블 로직 설계 🛡️
* **Problem**: 스포일러 방어 시스템의 특성상, 정상 리뷰를 오탐지하는 것보다 단 1개의 스포일러를 놓치는 것(미탐지)이 서비스에 훨씬 치명적임. 평균값(Soft Voting) 채택 시 모델 간 점수 희석 현상 발생.
* **Solution**: 세 모델 중 하나라도 강한 경고를 보내면 무조건 차단하는 **최대값 채택(Max Voting)** 방식 도입. 추가로 판정 임계값(Threshold)을 0.5에서 0.4로 하향 조정하여 **스포일러 재현율(Recall) 91%** 달성.

### 2️⃣ 문맥 보존을 위한 Soft Preprocessing 텍스트 정제 🧹
* **Problem**: 초기 전처리 단계에서 특수기호를 일괄 삭제(Hard Cleansing)했으나, "아니 주인공이 죽다니 ㅠㅠ"와 같은 리뷰에서 감정 기호('ㅠㅠ', '...')가 사라져 모델의 문맥 이해도가 급락하는 현상 발견.
* **Solution**: 한글, 영문뿐만 아니라 문맥 파악에 필수적인 구두점(.?!)과 자음/모음(ㅋㅋ, ㅠㅠ)을 보존하는 정규표현식(`soft_clean_text`) 파이프라인으로 전면 수정하여 딥러닝 모델의 은유적 스포일러 탐지력 향상.

### 3️⃣ LLM 의사 레이블링(Pseudo-Labeling) 비용 혁신 🤖
* **Problem**: 10만 건 이상의 텍스트 데이터에 정답(Label)을 수작업으로 다는 것은 시간적/물리적으로 불가능함.
* **Solution**: 오픈소스 LLM(Gemma-7B)에 최적화된 프롬프트를 주입하여 5,000건의 고품질 Gold Standard 라벨링을 자동화. 이후 머신러닝을 활용한 의사 레이블링으로 10만 건까지 데이터를 안전하게 확장하며 **데이터 구축 시간 90% 이상 절감**.

---

## 📁 Presentation & Documents
프로젝트의 기획부터 모델 성능 평가까지 정리된 상세 자료입니다.

* 📄 [프로젝트 최종 발표 자료 (PPT)](docs/ott_spoiler_detector_presentation.pdf)
* 📊 [독립 검증용 시험지 (200건 CSV)](data/test_set_clean_200.csv)

---

## 📂 Project Structure
```text
├── data/                       # 시연 및 테스트용 데이터 폴더
│   └── test_set_clean_200.csv  # 데모 시연에 사용된 200건의 기출문제 데이터
├── docs/                       # 발표 자료 및 시스템 아키텍처 이미지
├── notebooks/                  # 데이터 전처리 및 모델 학습 주피터 노트북
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_gemma_pseudo_labeling.ipynb
│   ├── 3_data_expansion_100k.ipynb
│   └── 4_ensemble_model_training.ipynb
├── src/                        # 실제 서비스 구동 핵심 코드
│   └── app.py                  # Streamlit 웹 대시보드 실행 코드
├── .gitignore                  # 딥러닝 모델 등 대용량 파일 업로드 제외 설정
├── requirements.txt            # 프로젝트 구동 필수 라이브러리 목록
└── README.md                   # 프로젝트 상세 소개서
