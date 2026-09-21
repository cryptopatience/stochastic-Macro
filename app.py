"""
==============================================================================
통합 대시보드 — SSO 전략 + 매크로 신용위험
실행: streamlit run app.py
==============================================================================
"""

import streamlit as st
from dashboard_auth import require_login
from dashboard_ui import apply_dashboard_style, dashboard_header, dashboard_card, theme_text, dashboard_plotly_chart

st.set_page_config(
    page_title="통합 투자 대시보드",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_dashboard_style()


# ─────────────────────────────────────────────────────────────────────────────
# 비밀번호 인증
# ─────────────────────────────────────────────────────────────────────────────
require_login()

dashboard_header("투자 리서치 대시보드", "기술적 신호부터 매크로 환경까지, 투자 판단에 필요한 정보를 한곳에서 확인하세요.")
st.markdown("""
<style>
    .desk-card {
        box-sizing: border-box;
        height: 340px;
        display: flex;
        flex-direction: column;
    }
    .desk-card .desk-value {
        font-size: 1.75rem;
        line-height: 1.35;
        min-height: 2.7em;
        word-break: keep-all;
        overflow-wrap: anywhere;
    }
    .desk-card .desk-note { margin-top: 16px; }
    @media (max-width: 700px) {
        .desk-card { height: auto; min-height: 260px; }
        .desk-card .desk-value { min-height: 0; }
    }
</style>
""", unsafe_allow_html=True)
for column, label, title, note, page in zip(
    st.columns(3),
    ["01 / TECHNICAL", "02 / MACRO", "03 / INTELLIGENCE"],
    ["SSO 전략", "매크로 신용위험", "AI 종합분석"],
    ["종목별 가격 흐름, 스토캐스틱 신호와 백테스트 성과를 확인합니다.",
     "경제지표, 수익률 곡선과 신용 환경을 통해 시장 위험을 살펴봅니다.",
     "기술적 분석과 매크로 데이터를 종합한 Gemini 분석을 확인합니다."],
    ["pages/1_SSO.py", "pages/2_Macro.py", "pages/3_AI_종합분석.py"],
):
    with column:
        dashboard_card(label, title, note)
        st.page_link(page, label="대시보드 열기 →", use_container_width=True)
st.caption("분석 결과는 선택한 기간과 설정에 따라 달라집니다. 각 화면에서 데이터 기준일을 확인하세요.")
