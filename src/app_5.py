import streamlit as st
import torch
import torch.nn as nn
import joblib
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. 1D-CNN 모델 뼈대 준비
# ==========================================
class SpoilerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 7500, 32)
        self.fc2 = nn.Linear(32, 1)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(0.5)
    def forward(self, x):
        x = self.relu(self.conv1(x.unsqueeze(1)))
        x = self.dropout(self.relu(self.fc1(self.pool(x).view(x.size(0), -1))))
        return self.fc2(x)

# ==========================================
# 2. 인공지능 뇌(모델) 불러오기 함수 (캐싱)
# ==========================================
@st.cache_resource
def load_ensemble_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr_model = joblib.load('ensemble_lr_model.pkl')
    tfidf_vec = joblib.load('ensemble_tfidf_vec.pkl')
    cnn_model = SpoilerCNN().to(device)
    cnn_model.load_state_dict(torch.load('ensemble_cnn_model.pth', map_location=device))
    cnn_model.eval()
    count_vec = joblib.load('ensemble_count_vec.pkl')
    tokenizer = AutoTokenizer.from_pretrained('./ensemble_electra_model')
    electra_model = AutoModelForSequenceClassification.from_pretrained('./ensemble_electra_model').to(device)
    electra_model.eval()
    return device, lr_model, tfidf_vec, cnn_model, count_vec, tokenizer, electra_model

device, lr_model, tfidf_vec, cnn_model, count_vec, tokenizer, electra_model = load_ensemble_models()

# ==========================================
# 3. Streamlit 웹 화면 UI 구성 (탭 분리)
# ==========================================
st.set_page_config(page_title="영화 스포일러 방어 시스템", page_icon="🛡️", layout="wide")

st.title("🛡️ 3대장 앙상블 스포일러 판독기")
st.markdown("**머신러닝(키워드) + 1D-CNN(패턴) + KcELECTRA(문맥)**가 결합된 철통 방어 시스템입니다.")

# 탭 생성 (2개의 화면으로 깔끔하게 분리)
tab1, tab2 = st.tabs(["📝 단일 리뷰 정밀 분석", "📁 대량 리뷰 실시간 처리 (Batch)"])

# ---------------------------------------------------------
# [탭 1] 단일 리뷰 분석 (게이지 바)
# ---------------------------------------------------------
with tab1:
    user_review = st.text_area("검사할 영화 리뷰를 입력하세요:", placeholder="리뷰를 입력하면 3개의 AI가 분석을 시작합니다.")

    if st.button("스포일러 판독 시작", type="primary"):
        if user_review.strip() == "":
            st.warning("리뷰를 입력해 주세요!")
        else:
            with st.spinner("3개의 인공지능이 리뷰를 분석 중입니다..."):
                # 예측 진행
                X_tfidf = tfidf_vec.transform([user_review])
                prob_lr = lr_model.predict_proba(X_tfidf)[0][1]
                
                X_cnn = torch.tensor(count_vec.transform([user_review]).toarray(), dtype=torch.float32).to(device)
                with torch.no_grad():
                    prob_cnn = torch.sigmoid(cnn_model(X_cnn)).item()
                    
                inputs = tokenizer(user_review, return_tensors="pt", truncation=True, max_length=128, padding=True).to(device)
                with torch.no_grad():
                    outputs = electra_model(**inputs)
                    prob_electra = torch.sigmoid(outputs.logits).item()
                    
                # 앙상블 투표 (최대값 채택 - 재현율 극대화)
                final_prob = max(prob_lr, prob_cnn, prob_electra)
                
                st.divider()
                
                # 시각적 게이지 바 (Progress Bar)
                st.subheader("📊 인공지능 엔진별 위험도 게이지")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.caption("1. 키워드 분석 (ML)")
                    st.progress(prob_lr)
                    st.write(f"**{prob_lr*100:.1f}%**")
                with col2:
                    st.caption("2. 패턴 분석 (CNN)")
                    st.progress(prob_cnn)
                    st.write(f"**{prob_cnn*100:.1f}%**")
                with col3:
                    st.caption("3. 문맥 분석 (ELECTRA)")
                    st.progress(prob_electra)
                    st.write(f"**{prob_electra*100:.1f}%**")
                
                st.write("---")
                
                # 최종 결과 출력 (핵심 단어 추출 제거됨)
                if final_prob >= 0.4:
                    st.error(f"🚨 **[차단됨] 스포일러가 포함되어 있습니다. (최대 위험도: {final_prob*100:.1f}%)**")
                else:
                    st.success(f"✅ **[통과] 안전한 리뷰입니다. (최대 위험도: {final_prob*100:.1f}%)**")

# ---------------------------------------------------------
# [탭 2] 대량 리뷰 실시간 처리 (CSV 업로드 + 파이 차트)
# ---------------------------------------------------------
with tab2:
    st.markdown("실제 서버 환경을 가정하여 **여러 개의 리뷰를 동시에 판독**합니다. (CSV 파일 업로드)")
    uploaded_file = st.file_uploader("리뷰가 담긴 CSV 파일을 올려주세요 (열 이름: 'text')", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("CSV 파일에 'text'라는 이름의 열(Column)이 필요합니다!")
        else:
            if st.button("대량 데이터 일괄 판독"):
                with st.spinner("앙상블 엔진이 수많은 리뷰를 고속으로 스캔하고 있습니다..."):
                    texts = df['text'].astype(str).values.tolist()
                    
                    # 빠른 예측을 위해 가장 가벼운 머신러닝(LR)을 메인 필터로 사용하는 시뮬레이션
                    X_tfidf_batch = tfidf_vec.transform(texts)
                    probs_batch = lr_model.predict_proba(X_tfidf_batch)[:, 1]
                    
                    df['위험도'] = np.round(probs_batch * 100, 1)
                    df['판정결과'] = ["🚨 스포일러" if p >= 0.4 else "✅ 안전" for p in probs_batch]
                    
                    # 파이 차트 그리기
                    st.subheader("📈 실시간 필터링 결과 대시보드")
                    spoiler_count = sum(df['판정결과'] == "🚨 스포일러")
                    safe_count = len(df) - spoiler_count
                    
                    col_chart, col_data = st.columns([1, 2])
                    
                    with col_chart:
                        # Streamlit에 내장된 파이 차트 라이브러리 활용
                        import plotly.express as px
                        fig = px.pie(values=[safe_count, spoiler_count], names=['안전 통과', '스포일러 차단'], 
                                     color_discrete_sequence=['#10b981', '#ef4444'], hole=0.4)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col_data:
                        st.dataframe(df[['text', '위험도', '판정결과']], use_container_width=True)