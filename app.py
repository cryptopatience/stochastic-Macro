"""
==============================================================================
통합 대시보드 — SSO 전략 + 매크로 신용위험
실행: streamlit run app.py
==============================================================================
"""

import streamlit as st

st.set_page_config(
    page_title="통합 투자 대시보드",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    section[data-testid="stSidebar"] { background-color: #f5f5f5; }
    h1, h2, h3 { color: #1a1a1a !important; }
    hr { border-color: #d0d0d0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 비밀번호 인증
# ─────────────────────────────────────────────────────────────────────────────
_CORRECT_PW = st.secrets.get("APP_PASSWORD", "1234")

if not st.session_state.get("authenticated"):
    st.markdown("## 🔒 통합 투자 대시보드")
    st.markdown("접속하려면 비밀번호를 입력하세요.")
    pw = st.text_input("비밀번호", type="password", key="pw_input")
    if st.button("로그인", type="primary"):
        if pw == _CORRECT_PW:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ 비밀번호가 틀렸습니다.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 인증 후 메인 화면
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    if st.button("🔓 로그아웃", key="logout_btn"):
        st.session_state["authenticated"] = False
        st.rerun()

st.markdown("# 🏦 통합 투자 대시보드")
st.markdown("""
왼쪽 사이드바에서 원하는 페이지를 선택하세요.

| 페이지 | 설명 |
|--------|------|
| 📈 **SSO 전략** | 슬로우 스토캐스틱 오실레이터 전략 · Mag 7 신호 · 백테스트 |
| 🏦 **매크로 신용위험** | FRED 경제지표 · 수익률 곡선 · 연체율 · Gemini AI 분석 |
| 🤖 **AI 종합분석** | SSO + 매크로 통합 딥다이브 · Discord 자동 발송 |
""")
