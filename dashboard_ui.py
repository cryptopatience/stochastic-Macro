"""Shared presentation for the investment dashboard."""
from html import escape

import streamlit as st

LIGHT_COLORS = {
    "#090e15": "#f6f8fc", "#0e151f": "#edf1f7", "#101823": "#ffffff",
    "#e6edf5": "#172438", "#f0f4fa": "#172438", "#263140": "#d5ddea",
    "#a7b4c6": "#526176", "#b7c6db": "#40536c", "#334156": "#c4cfdf",
    "#edc879": "#805b15", "#d8ad55": "#946b19", "#131d2a": "#f0f4fa",
    "#12332d": "#e2f3eb", "#392027": "#fbe8e9", "#352c1b": "#fff2d4",
    "#e3b341": "#805b15", "#a5d6ff": "#155fa0",
}


def theme_text(value):
    if st.session_state.get("dashboard_theme", "Navy") == "Light":
        for dark, light in LIGHT_COLORS.items():
            value = value.replace(dark, light)
    return value


def _save_theme():
    st.session_state["dashboard_theme"] = st.session_state["_dashboard_theme_picker"]


def dashboard_plotly_chart(figure, **kwargs):
    light = st.session_state.get("dashboard_theme", "Navy") == "Light"
    figure.update_layout(
        dragmode="zoom",
        template="plotly_white" if light else "plotly_dark",
        paper_bgcolor=theme_text("#101823"), plot_bgcolor=theme_text("#101823"),
        font_color=theme_text("#e6edf5"),
        legend=dict(bgcolor=theme_text("#101823"), font_color=theme_text("#e6edf5")),
        hoverlabel=dict(bgcolor=theme_text("#101823"), font_color=theme_text("#e6edf5")),
    )
    figure.update_xaxes(gridcolor=theme_text("#263140"), fixedrange=False, automargin=True)
    figure.update_yaxes(gridcolor=theme_text("#263140"), fixedrange=False, automargin=True)
    kwargs["config"] = {
        "scrollZoom": True,
        "showAxisDragHandles": True,
        "showAxisRangeEntryBoxes": True,
        "showTips": True,
        "displayModeBar": True,
        "displaylogo": False,
        "doubleClick": "reset",
        "modeBarButtonsToAdd": ["zoomIn2d", "zoomOut2d", "resetScale2d"],
        **kwargs.get("config", {}),
    }
    kwargs["theme"] = None
    chart = st.plotly_chart(figure, **kwargs)
    st.caption("축 늘리기·줄이기: 축 양 끝의 숫자 근처를 클릭한 채 드래그하세요. 가로축은 좌우, 세로축은 위아래로 움직입니다. 축 가운데는 범위 이동 · 그래프 더블클릭은 원래 범위로 복원")
    return chart


def apply_dashboard_style():
    st.session_state.setdefault("dashboard_theme", "Navy")
    st.session_state["_dashboard_theme_picker"] = st.session_state["dashboard_theme"]
    st.sidebar.radio("화면 테마", ["Navy", "Light"], horizontal=True,
                     key="_dashboard_theme_picker", on_change=_save_theme)
    st.markdown(theme_text("""
<style>
    .stApp { background: #090e15; color: #e6edf5; }
    [data-testid="stHeader"] { background: #090e15; }
    .block-container { max-width: 1600px; padding-top: 2rem; padding-bottom: 3rem; }
    section[data-testid="stSidebar"] { background: #0e151f; border-right: 1px solid #263140; }
    h1, h2, h3, h4 { color: #e6edf5 !important; letter-spacing: -.025em; }
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.35rem !important; }
    h3 { font-size: 1.1rem !important; }
    hr { border-color: #263140; margin: 1.2rem 0; }
    [data-testid="stMetric"] { background: #101823; border: 1px solid #263140;
        border-radius: 10px; padding: 18px; min-height: 125px; border-top: 2px solid #d8ad55; }
    [data-testid="stMetricLabel"] { color: #a7b4c6; font-size: .8rem; }
    [data-testid="stMetricValue"] { color: #e6edf5; font-variant-numeric: tabular-nums; }
    [data-testid="stPlotlyChart"] { border: 1px solid #263140; border-radius: 10px;
        overflow: hidden; background: #101823; }
    [data-testid="stExpander"] { border-color: #263140; background: #101823; }
    button[data-baseweb="tab"] { color: #a7b4c6; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #edc879; }
    .desk-header { display: flex; justify-content: space-between; align-items: center;
        gap: 20px; padding: 4px 0 22px; border-bottom: 1px solid #263140; margin-bottom: 24px; }
    .desk-brand { color: #edc879; font-size: .75rem; letter-spacing: .18em; font-weight: 700; }
    .desk-title { color: #f0f4fa; font-size: 1.7rem; font-weight: 700; margin: 8px 0; }
    .desk-sub { color: #a7b4c6; font-size: .85rem; line-height: 1.6; }
    .desk-badge { color: #b7c6db; border: 1px solid #334156; padding: 8px 14px;
        border-radius: 6px; font-size: .75rem; white-space: nowrap; }
    .desk-card { background: #101823; border: 1px solid #263140; border-radius: 10px;
        padding: 22px; margin-bottom: 12px; }
    .desk-label { color: #a7b4c6; font-size: .75rem; letter-spacing: .08em; margin-bottom: 12px; }
    .desk-value { font-size: 2rem; color: #edc879; font-weight: 700; font-variant-numeric: tabular-nums; }
    .desk-note { color: #a7b4c6; font-size: .85rem; margin-top: 10px; line-height: 1.7; }
    [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
    [data-testid="stCaptionContainer"], [data-testid="stSidebarNav"] a,
    [data-testid="stPageLink"] a { color: #e6edf5; }
    [data-baseweb="input"], [data-baseweb="base-input"],
    [data-baseweb="select"] > div, [data-baseweb="textarea"],
    [data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"],
    [role="option"], [data-baseweb="calendar"] {
        background-color: #101823 !important; color: #e6edf5 !important;
    }
    input, textarea, [data-baseweb="select"] span { color: #e6edf5 !important; -webkit-text-fill-color: #e6edf5; }
    input::placeholder, textarea::placeholder { color: #a7b4c6 !important; }
    [data-testid="stBaseButton-secondary"], [data-testid="stBaseButton-headerNoPadding"],
    [data-testid="stBaseButton-header"] {
        background: #101823; color: #e6edf5; border-color: #263140;
    }
    [data-testid="stBaseButton-primary"] { background: #d8ad55; color: #090e15; border: 0; }
    [data-testid="stExpander"] summary { color: #e6edf5; }
    [data-testid="stMetricDelta"] { color: #a7b4c6; }
    @media(max-width: 700px) {
        .desk-header { align-items: flex-start; flex-direction: column; }
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .desk-title { font-size: 1.4rem; }
    }
</style>
"""), unsafe_allow_html=True)


def dashboard_header(title, subtitle, badge="MARKET RESEARCH"):
    st.markdown(f"""<div class="desk-header"><div>
    <div class="desk-brand">SCGT / INVESTMENT INTELLIGENCE</div>
    <div class="desk-title">{escape(title)}</div>
    <div class="desk-sub">{escape(subtitle)}</div></div>
    <div class="desk-badge">{escape(badge)}</div></div>""", unsafe_allow_html=True)


def dashboard_card(label, value, note):
    st.markdown(f"""<div class="desk-card"><div class="desk-label">{escape(label)}</div>
    <div class="desk-value">{escape(value)}</div>
    <div class="desk-note">{escape(note)}</div></div>""", unsafe_allow_html=True)
