"""
==============================================================================
SSO 슬로우 스토캐스틱 오실레이터 — Streamlit 인터랙티브 대시보드
실행: streamlit run sso_dashboard.py
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import streamlit as st
import requests
from datetime import datetime
import google.generativeai as genai

# ── 인증 체크 ─────────────────────────────────────────────────────────────────
if not st.session_state.get("authenticated"):
    st.error("🔒 접근 권한이 없습니다. 메인 페이지에서 로그인하세요.")
    st.stop()

# ── AI 결과 디스크 캐시 헬퍼 ──────────────────────────────────────────────────
import json as _json, os as _os_cache

_AI_CACHE_DIR = _os_cache.path.join(_os_cache.path.dirname(_os_cache.path.abspath(__file__)), "..", "_ai_cache")
_os_cache.makedirs(_AI_CACHE_DIR, exist_ok=True)

def _ai_cache_load(key: str):
    path = _os_cache.path.join(_AI_CACHE_DIR, f"{key}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None

def _ai_cache_save(key: str, data: dict):
    path = _os_cache.path.join(_AI_CACHE_DIR, f"{key}.json")
    try:
        safe = {k: v for k, v in data.items() if k != "scan_df"}
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(safe, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# 새로고침 후 복원
if "sso_ai_result" not in st.session_state:
    _cached = _ai_cache_load("sso_ai_result")
    if _cached:
        st.session_state["sso_ai_result"] = _cached

# ─────────────────────────────────────────────────────────────────────────────
# 라이트 테마 CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    section[data-testid="stSidebar"] { background-color: #f5f5f5; }

    /* 메트릭 카드 */
    div[data-testid="metric-container"] {
        background-color: #f5f5f5;
        border: 1px solid #d0d0d0;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="metric-container"] label { color: #555555 !important; font-size: 0.78rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #1a1a1a !important; font-size: 1.4rem; font-weight: 700;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] svg { display: none; }

    /* 테이블 */
    .stDataFrame { border-radius: 8px; overflow: hidden; }

    /* 제목 */
    h1, h2, h3 { color: #1a1a1a !important; }
    .subtitle  { color: #555555; font-size: 0.9rem; margin-top: -10px; }

    /* 구분선 */
    hr { border-color: #d0d0d0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 핵심 함수들
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def download_data(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if df.empty:
        return df
    # MultiIndex 컬럼 처리 (yfinance 버전 따라 다름)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    else:
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # 필요한 컬럼만 추출
    available = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    if "Close" not in available:
        return pd.DataFrame()
    df = df[available].dropna()
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def calc_sso(df: pd.DataFrame, k_period: int, k_smooth: int, d_smooth: int) -> pd.DataFrame:
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    k_fast = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)
    df = df.copy()
    df["%K"] = k_fast.rolling(k_smooth).mean()
    df["%D"] = df["%K"].rolling(d_smooth).mean()
    return df


def detect_signals(df: pd.DataFrame, ob: float, os_: float) -> pd.DataFrame:
    df = df.copy()
    K, D = df["%K"].values, df["%D"].values
    n = len(df)
    first_buy  = np.zeros(n, dtype=bool)
    first_sell = np.zeros(n, dtype=bool)
    second_buy  = np.zeros(n, dtype=bool)
    second_sell = np.zeros(n, dtype=bool)

    buy_first_done = sell_first_done = False

    for i in range(1, n):
        if np.isnan(K[i]) or np.isnan(D[i]):
            continue

        # ── 매수 (과매도 구간 골든 크로스) ──────────────────────────────────
        k_up = K[i-1] < D[i-1] and K[i] >= D[i]
        if k_up and (K[i] < os_ + 10):
            if not buy_first_done:
                first_buy[i] = True
                buy_first_done = True
            else:
                second_buy[i] = True
                buy_first_done = False
        if K[i] > os_ + 15 and D[i] > os_ + 15:
            buy_first_done = False

        # ── 매도 (과매수 구간 데드 크로스) ──────────────────────────────────
        k_dn = K[i-1] > D[i-1] and K[i] <= D[i]
        if k_dn and (K[i] > ob - 10):
            if not sell_first_done:
                first_sell[i] = True
                sell_first_done = True
            else:
                second_sell[i] = True
                sell_first_done = False
        if K[i] < ob - 15 and D[i] < ob - 15:
            sell_first_done = False

    df["first_buy"]   = first_buy
    df["first_sell"]  = first_sell
    df["second_buy"]  = second_buy
    df["second_sell"] = second_sell
    return df


def run_backtest(
    df: pd.DataFrame,
    hold_days: int,
    total_capital: float,
    risk_per_trade: float,
    stop_loss_pct: float,
    partial_exit_pct: float,
    target_profit_pct: float,
) -> pd.DataFrame:
    trades = []
    close  = df["Close"].values
    high   = df["High"].values
    low    = df["Low"].values
    dates  = df.index
    idx    = df.index.get_indexer(df[df["second_buy"]].index)

    remaining_capital = total_capital

    for ei in idx:
        if ei < 0:
            continue

        buy_i       = ei + 1 if ei + 1 < len(close) else ei
        entry_price = float(close[buy_i])
        stop_price  = entry_price * (1 - stop_loss_pct / 100)
        target_price = entry_price * (1 + target_profit_pct / 100)

        # ── 포지션 사이즈 (리스크 기반) ──────────────────────────────
        risk_amount    = remaining_capital * (risk_per_trade / 100)
        risk_per_share = entry_price - stop_price
        shares         = risk_amount / risk_per_share if risk_per_share > 0 else 0
        position_value = shares * entry_price
        if position_value > remaining_capital:
            shares         = remaining_capital / entry_price
            position_value = remaining_capital
        shares = int(shares)
        if shares == 0:
            continue

        partial_shares = int(shares * partial_exit_pct / 100)
        remain_shares  = shares - partial_shares

        # ── 보유 기간 중 가격 경로 확인 ──────────────────────────────
        exit_idx     = min(buy_i + hold_days, len(close) - 1)
        hit_stop     = False
        hit_target   = False
        stop_exit_i  = exit_idx
        target_exit_i = exit_idx

        for j in range(buy_i + 1, exit_idx + 1):
            if not hit_stop and float(low[j]) <= stop_price:
                hit_stop    = True
                stop_exit_i = j
            if not hit_target and float(high[j]) >= target_price:
                hit_target    = True
                target_exit_i = j

        # ── 청산 결정 ────────────────────────────────────────────────
        if hit_stop and (not hit_target or stop_exit_i <= target_exit_i):
            exit_price_final = stop_price
            pnl              = (exit_price_final - entry_price) * shares
            result_label     = "🛑 손절"
            final_exit_i     = stop_exit_i
        elif hit_target and partial_exit_pct < 100:
            pnl_partial      = (target_price - entry_price) * partial_shares
            exit_price_full  = float(close[exit_idx])
            pnl_remain       = (exit_price_full - entry_price) * remain_shares
            pnl              = pnl_partial + pnl_remain
            exit_price_final = target_price
            result_label     = "🎯 목표 달성"
            final_exit_i     = exit_idx
        else:
            exit_price_final = float(close[exit_idx])
            pnl              = (exit_price_final - entry_price) * shares
            result_label     = "✅ 수익" if pnl > 0 else "❌ 손실"
            final_exit_i     = exit_idx

        ret_pct = (exit_price_final - entry_price) / entry_price * 100

        trades.append({
            "매수일":          dates[ei].strftime("%Y-%m-%d"),
            "매도일":          dates[final_exit_i].strftime("%Y-%m-%d"),
            "매수가":          round(entry_price, 2),
            "손절가":          round(stop_price, 2),
            "1차목표가":       round(target_price, 2),
            "매도가":          round(exit_price_final, 2),
            "주식 수":         shares,
            "투자금 ($)":      round(position_value, 0),
            "리스크 금액 ($)": round(risk_amount, 0),
            "손익 ($)":        round(pnl, 0),
            "수익률(%)":       round(ret_pct, 2),
            "결과":            result_label,
        })

    return pd.DataFrame(trades)


def build_chart(df: pd.DataFrame, ticker: str, ob: float, os_: float) -> go.Figure:
    sb = df[df["second_buy"]]
    ss = df[df["second_sell"]]
    fb = df[df["first_buy"]]
    fs = df[df["first_sell"]]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.28, 0.17],
        vertical_spacing=0.02,
        subplot_titles=(
            f"📊 {ticker} 가격 차트",
            "🔵 슬로우 스토캐스틱 오실레이터 (SSO)",
            "📦 거래량"
        )
    )

    # ── 캔들스틱 ──────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        name="캔들",
        increasing=dict(line=dict(color="#3fb950", width=1), fillcolor="#3fb950"),
        decreasing=dict(line=dict(color="#f85149", width=1), fillcolor="#f85149"),
    ), row=1, col=1)

    # 첫 번째 신호 (반투명, 저가/고가 아래위에 배치)
    if not fb.empty:
        fig.add_trace(go.Scatter(
            x=fb.index, y=fb["Low"] * 0.995, mode="markers", name="1st Buy (무시)",
            marker=dict(symbol="triangle-up", color="rgba(63,185,80,0.35)", size=9),
            showlegend=True
        ), row=1, col=1)
    if not fs.empty:
        fig.add_trace(go.Scatter(
            x=fs.index, y=fs["High"] * 1.005, mode="markers", name="1st Sell (무시)",
            marker=dict(symbol="triangle-down", color="rgba(248,81,73,0.35)", size=9),
            showlegend=True
        ), row=1, col=1)

    # 두 번째 신호 (선명, 저가/고가 아래위에 배치)
    if not sb.empty:
        fig.add_trace(go.Scatter(
            x=sb.index, y=sb["Low"] * 0.990, mode="markers", name="★ 2nd Buy",
            marker=dict(symbol="triangle-up", color="#3fb950", size=16,
                        line=dict(width=1.5, color="#ffffff")),
            showlegend=True
        ), row=1, col=1)
    if not ss.empty:
        fig.add_trace(go.Scatter(
            x=ss.index, y=ss["High"] * 1.010, mode="markers", name="★ 2nd Sell",
            marker=dict(symbol="triangle-down", color="#f85149", size=16,
                        line=dict(width=1.5, color="#ffffff")),
            showlegend=True
        ), row=1, col=1)

    # ── SSO ───────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df.index, y=df["%K"], name="%K Slow",
        line=dict(color="#e3b341", width=1.8)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["%D"], name="%D Slow",
        line=dict(color="#a5d6ff", width=1.2, dash="dot")
    ), row=2, col=1)

    # 과매수/과매도 기준선
    for level, color, label in [(ob, "#f85149", f"과매수 {int(ob)}"),
                                 (os_, "#3fb950", f"과매도 {int(os_)}")]:
        fig.add_hline(y=level, line_color=color, line_width=0.8,
                      line_dash="dot", row=2, col=1,
                      annotation_text=label,
                      annotation_font_color=color,
                      annotation_position="right")

    # 음영
    fig.add_hrect(y0=ob, y1=100, fillcolor="rgba(248,81,73,0.08)",
                  line_width=0, row=2, col=1)
    fig.add_hrect(y0=0, y1=os_, fillcolor="rgba(63,185,80,0.08)",
                  line_width=0, row=2, col=1)

    # SSO 위 신호 마커
    if not sb.empty:
        fig.add_trace(go.Scatter(
            x=sb.index, y=df.loc[sb.index, "%K"], mode="markers",
            marker=dict(symbol="triangle-up", color="#3fb950", size=11),
            showlegend=False
        ), row=2, col=1)
    if not ss.empty:
        fig.add_trace(go.Scatter(
            x=ss.index, y=df.loc[ss.index, "%K"], mode="markers",
            marker=dict(symbol="triangle-down", color="#f85149", size=11),
            showlegend=False
        ), row=2, col=1)

    # ── 거래량 ────────────────────────────────────────────────────────────────
    vol_colors = ["#3fb950" if c >= o else "#f85149"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="거래량",
        marker_color=vol_colors, opacity=0.6, showlegend=False
    ), row=3, col=1)

    # ── 레이아웃 ──────────────────────────────────────────────────────────────
    fig.update_layout(
        height=750,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#1a1a1a", family="monospace"),
        legend=dict(bgcolor="#f5f5f5", bordercolor="#d0d0d0",
                    borderwidth=1, font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=60, r=40, t=60, b=40),
    )
    fig.update_xaxes(gridcolor="#e0e0e0", showgrid=True)
    fig.update_yaxes(gridcolor="#e0e0e0", showgrid=True)
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Discord 알림 / Mag 7 스캔
# ─────────────────────────────────────────────────────────────────────────────
MAG7 = {
    "AAPL":    "🍎 Apple",
    "MSFT":    "🪟 Microsoft",
    "NVDA":    "💚 NVIDIA",
    "GOOGL":   "🔍 Alphabet",
    "AMZN":    "📦 Amazon",
    "META":    "👁️ Meta",
    "TSLA":    "⚡ Tesla",
    "BTC-USD": "₿ Bitcoin",
}


def send_discord(webhook_url: str, embeds: list) -> bool:
    """Discord Webhook으로 embed 메시지 전송."""
    try:
        resp = requests.post(
            webhook_url,
            json={"embeds": embeds},
            timeout=10,
        )
        return resp.status_code in (200, 204)
    except Exception:
        return False


def build_embed(ticker: str, signal: str, row: pd.Series, interval: str) -> dict:
    """Discord embed 딕셔너리 생성."""
    is_buy   = "buy" in signal
    color    = 0x3FB950 if is_buy else 0xF85149   # green / red
    label    = "★ 2번째 매수 신호 🟢" if is_buy else "★ 2번째 매도 신호 🔴"
    tf_label = "일봉" if interval == "1d" else "주봉"
    ts       = str(row.name)[:16]
    return {
        "title":       f"{MAG7.get(ticker, ticker)}  ({ticker}) — {label}",
        "color":       color,
        "description": f"**타임프레임**: {tf_label}\n**신호 발생 시점**: {ts}",
        "fields": [
            {"name": "종가",  "value": f"`{row['Close']:.2f}`",  "inline": True},
            {"name": "%K",   "value": f"`{row['%K']:.2f}`",     "inline": True},
            {"name": "%D",   "value": f"`{row['%D']:.2f}`",     "inline": True},
        ],
        "footer": {"text": f"SSO 전략 대시보드 · {datetime.now().strftime('%Y-%m-%d %H:%M')}"},
    }


import os as _os
_DAILY_REPORT_STATE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".daily_report_sent")


def build_daily_report_embeds(alerts: list, interval: str) -> list:
    """Mag 7 전체 현황을 담은 일일 리포트 Discord embed 목록 생성."""
    tf_label = "일봉" if interval == "1d" else "주봉"
    today    = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── 요약 embed ─────────────────────────────────────────────────────────────
    signal_lines = []
    for a in alerts:
        if a["signal"] == "second_buy":
            icon = "🟢 2nd Buy"
        elif a["signal"] == "second_sell":
            icon = "🔴 2nd Sell"
        else:
            icon = "─"
        signal_lines.append(
            f"**{a['name']}** ({a['ticker']})  |  종가 `{a['price']}`"
            f"  %K `{a['k_now']}`  %D `{a['d_now']}`  →  {icon}"
        )

    # ── 최근 1개월 신호 요약 ───────────────────────────────────────────────────
    recent_lines = []
    for a in alerts:
        for rs in a["recent_signals"]:
            icon = "🟢 2nd Buy" if rs["signal"] == "second_buy" else "🔴 2nd Sell"
            recent_lines.append(
                f"**{a['ticker']}**  {rs['date']}  종가 `{rs['price']}`  → {icon}"
            )

    embeds = [
        {
            "title":       f"📊 Mag 7 일일 SSO 리포트 — {today}",
            "color":       0x58A6FF,
            "description": f"**타임프레임**: {tf_label}\n\n"
                           + "\n".join(signal_lines),
            "footer":      {"text": "SSO 전략 대시보드 · 일일 자동 리포트"},
        }
    ]
    if recent_lines:
        embeds.append({
            "title":       "📅 최근 1개월 2nd 신호 발생 내역",
            "color":       0xE3B341,
            "description": "\n".join(recent_lines),
        })
    return embeds


def check_and_send_daily_report(wh_url: str, interval: str,
                                 k_period: int, k_smooth: int, d_smooth: int,
                                 ob: float, os_: float,
                                 report_hour: int, report_minute: int) -> str:
    """
    오늘 지정 시각 이후이고 아직 전송하지 않았으면 일일 리포트를 Discord로 전송.
    반환: "sent" | "already_sent" | "not_yet" | "error"
    """
    now       = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # 이미 오늘 전송했는지 확인
    if _os.path.exists(_DAILY_REPORT_STATE):
        with open(_DAILY_REPORT_STATE, encoding="utf-8") as f:
            if f.read().strip() == today_str:
                return "already_sent"

    # 전송 시각 미도달
    if (now.hour, now.minute) < (report_hour, report_minute):
        return "not_yet"

    # 전송
    try:
        alerts = scan_mag7(interval, k_period, k_smooth, d_smooth, ob, os_, _tickers=tuple(MAG7.keys()))
        embeds = build_daily_report_embeds(alerts, interval)
        ok = False
        for i in range(0, len(embeds), 10):
            ok = send_discord(wh_url, embeds[i:i + 10])
        if ok:
            with open(_DAILY_REPORT_STATE, "w", encoding="utf-8") as f:
                f.write(today_str)
            return "sent"
        return "error"
    except Exception:
        return "error"


@st.cache_data(show_spinner=False, ttl=300)
def scan_mag7(interval: str, k_period: int, k_smooth: int,
              d_smooth: int, ob: float, os_: float,
              _tickers: tuple = tuple(MAG7.keys())) -> list:
    """
    Mag 7 전 종목 최신 데이터 스캔.
    반환: [{"ticker", "name", "signal", "row", "k_now", "d_now", "price",
             "recent_signals": [{"signal", "date", "price", "k", "d"}, ...]}, ...]
    """
    from datetime import timedelta
    end   = datetime.today()
    # 충분한 봉 수 확보를 위해 넉넉히 90일(일봉) 또는 3년(주봉)
    start = end - timedelta(days=90 if interval == "1d" else 1095)
    cutoff = end - timedelta(days=30)
    alerts = []

    for ticker in MAG7:
        try:
            df = download_data(ticker, start.strftime("%Y-%m-%d"),
                               end.strftime("%Y-%m-%d"), interval)
            if df.empty or len(df) < k_period + k_smooth + d_smooth:
                continue
            df = calc_sso(df, k_period, k_smooth, d_smooth)
            df = detect_signals(df, ob, os_)

            last = df.dropna(subset=["%K", "%D"]).iloc[-1]

            # 마지막 봉 신호
            signal = None
            if last["second_buy"]:
                signal = "second_buy"
            elif last["second_sell"]:
                signal = "second_sell"

            # 최근 1개월 내 2번째 신호 목록
            recent = []
            df_valid = df.dropna(subset=["%K", "%D"])
            df_recent = df_valid[df_valid.index >= pd.Timestamp(cutoff)]
            for dt, row in df_recent.iterrows():
                if row["second_buy"]:
                    recent.append({"signal": "second_buy",  "date": str(dt)[:10],
                                   "price": round(float(row["Close"]), 2),
                                   "k": round(float(row["%K"]), 2),
                                   "d": round(float(row["%D"]), 2)})
                elif row["second_sell"]:
                    recent.append({"signal": "second_sell", "date": str(dt)[:10],
                                   "price": round(float(row["Close"]), 2),
                                   "k": round(float(row["%K"]), 2),
                                   "d": round(float(row["%D"]), 2)})

            alerts.append({
                "ticker":         ticker,
                "name":           MAG7[ticker],
                "signal":         signal,
                "row":            last,
                "k_now":          round(float(last["%K"]), 2),
                "d_now":          round(float(last["%D"]), 2),
                "price":          round(float(last["Close"]), 2),
                "date":           str(last.name)[:10],
                "recent_signals": recent,
            })
        except Exception:
            # 데이터 수집 실패 시 빈 항목으로 포함 (누락 방지)
            alerts.append({
                "ticker":         ticker,
                "name":           MAG7[ticker],
                "signal":         None,
                "row":            None,
                "k_now":          None,
                "d_now":          None,
                "price":          None,
                "date":           "N/A",
                "recent_signals": [],
            })

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# 사이드바 — 파라미터 설정
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 전략 파라미터")
    st.markdown("---")

    ticker = st.text_input("종목 티커", value="AAPL",
                           help="예) AAPL, TSLA, MSFT, 005930.KS (삼성전자)").upper()

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("시작일", value=pd.Timestamp("2022-01-01"))
    with col2:
        end_date = st.date_input("종료일", value=pd.Timestamp.today())

    st.markdown("---")
    st.markdown("#### 📐 SSO 파라미터")

    k_period = st.slider("%K 기간 (관찰 기간)", min_value=5, max_value=30,
                         value=9, step=1,
                         help="권장값: 9 — 단기 사이클 절반 길이에 최적")
    k_smooth = st.slider("%K Smooth (이동평균)", min_value=2, max_value=10,
                         value=5, step=1)
    d_smooth = st.slider("%D Smooth (시그널선)", min_value=2, max_value=10,
                         value=3, step=1)

    st.markdown("---")
    st.markdown("#### 🎯 과매수 / 과매도 기준")
    overbought = st.slider("과매수 기준", min_value=60, max_value=90,
                           value=80, step=5)
    oversold   = st.slider("과매도 기준", min_value=10, max_value=40,
                           value=20, step=5)

    st.markdown("---")
    st.markdown("#### 🔄 백테스트 설정")
    hold_days = st.slider("보유 기간 (일)", min_value=5, max_value=60,
                          value=20, step=5,
                          help="두 번째 매수 신호 발생 후 보유할 거래일 수")

    st.markdown("---")
    st.markdown("#### 💰 자금 관리 (Position Sizing)")
    total_capital = st.number_input(
        "총 투자 가능 자본 (USD)",
        min_value=1000, max_value=10_000_000, value=10000, step=1000,
    )
    risk_per_trade = st.slider(
        "트레이드당 리스크 (%)",
        min_value=0.5, max_value=5.0, value=1.0, step=0.5,
        help="총 자본 대비 한 번 손절 시 허용 손실 비율",
    )
    stop_loss = st.slider(
        "손절 기준 (%)",
        min_value=1.0, max_value=15.0, value=5.0, step=0.5,
        help="매수가 대비 손절가 하락폭",
    )
    target_profit_pct = st.slider(
        "1차 목표 수익률 (%)",
        min_value=3, max_value=30, value=10, step=1,
    )
    partial_exit_pct = st.slider(
        "1차 목표 달성 시 청산 비율 (%)",
        min_value=25, max_value=100, value=50, step=25,
        help="목표 수익률 달성 시 먼저 매도할 비율",
    )

    st.markdown("---")
    st.markdown("#### ⏱️ 타임프레임 비교")
    compare_mode = st.checkbox("일봉 vs 주봉 비교 모드", value=True)

    st.session_state["discord_url"] = "https://discord.com/api/webhooks/1487415854839894076/i2HkxX91ZbcWFzOHe9QZjLvNNXPl-j6t1rZs2hnQcvC0gbzk0l0Ohyce2nXU5C3IYD0A"

    st.markdown("---")
    st.markdown("#### 🔔 Mag 7 디스코드 알림")
    mag7_interval = st.selectbox(
        "모니터링 타임프레임",
        ["1d", "1wk"],
        format_func=lambda x: "일봉 (1d)" if x == "1d" else "주봉 (1wk)",
    )
    auto_alert = st.toggle(
        "⚡ 페이지 로드 시 자동 알림",
        value=False,
        help="활성화 시 Discord URL이 있으면 페이지 열릴 때마다 신호 발생 종목을 자동 전송",
    )

    st.markdown("---")
    st.markdown("#### 📊 일일 리포트")
    daily_report_on = st.toggle("매일 자동 리포트 전송", value=True,
                                help="매일 오전 8:00에 Mag 7 전체 SSO 현황을 Discord로 전송")
    report_hour, report_minute = 8, 0

    mag7_btn = st.button("🔍 Mag 7 신호 점검", use_container_width=True)

    run_btn = st.button("🚀 분석 실행", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# 메인 타이틀
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
# 📈 슬로우 스토캐스틱 오실레이터 (SSO) 전략 대시보드
<p class='subtitle'>
두 번째 교차 신호(Second Signal) 기반 — 타임 사이클 동기화 원리 적용
</p>
""", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 실행 로직
# ─────────────────────────────────────────────────────────────────────────────
def load_and_process(ticker, start_date, end_date, interval, k_period, k_smooth, d_smooth,
                     overbought, oversold, hold_days,
                     total_capital, risk_per_trade, stop_loss_pct, partial_exit_pct, target_profit_pct):
    df_raw = download_data(ticker, str(start_date), str(end_date), interval)
    if df_raw.empty:
        return None, None
    df = calc_sso(df_raw, k_period, k_smooth, d_smooth)
    df = detect_signals(df, overbought, oversold)
    trades = run_backtest(df, hold_days, total_capital, risk_per_trade,
                          stop_loss_pct, partial_exit_pct, target_profit_pct)
    return df, trades


if run_btn or "df" not in st.session_state:
    with st.spinner(f"📡 {ticker} 데이터 불러오는 중..."):
        try:
            df, trades = load_and_process(
                ticker, start_date, end_date, "1d",
                k_period, k_smooth, d_smooth, overbought, oversold, hold_days,
                total_capital, risk_per_trade, stop_loss, partial_exit_pct, target_profit_pct,
            )
            if df is None:
                st.error("❌ 데이터를 불러올 수 없습니다. 티커를 확인해 주세요.")
                st.stop()

            st.session_state["df"]     = df
            st.session_state["trades"] = trades
            st.session_state["ticker"] = ticker

            if compare_mode:
                df_wk, trades_wk = load_and_process(
                    ticker, start_date, end_date, "1wk",
                    k_period, k_smooth, d_smooth, overbought, oversold, hold_days,
                    total_capital, risk_per_trade, stop_loss, partial_exit_pct, target_profit_pct,
                )
                st.session_state["df_wk"]     = df_wk
                st.session_state["trades_wk"] = trades_wk
            else:
                st.session_state.pop("df_wk", None)
                st.session_state.pop("trades_wk", None)

        except Exception as e:
            st.error(f"오류 발생: {e}")
            st.stop()

df     = st.session_state["df"]
trades = st.session_state["trades"]
ticker = st.session_state["ticker"]
df_wk     = st.session_state.get("df_wk")
trades_wk = st.session_state.get("trades_wk")

# ─────────────────────────────────────────────────────────────────────────────
# 현재 SSO 상태 배너
# ─────────────────────────────────────────────────────────────────────────────
last = df.dropna(subset=["%K", "%D"]).iloc[-1]
k_now, d_now = last["%K"], last["%D"]

if k_now > overbought:
    zone_label = "🔴 과매수 구간"
    zone_color = "#f85149"
elif k_now < oversold:
    zone_label = "🟢 과매도 구간"
    zone_color = "#3fb950"
else:
    zone_label = "⚪ 중립 구간"
    zone_color = "#555555"

st.markdown(f"""
<div style="background:#f5f5f5;border:1px solid {zone_color};border-radius:10px;
            padding:12px 20px;margin-bottom:16px;">
  <span style="color:{zone_color};font-weight:700;font-size:1.05rem;">{zone_label}</span>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <span style="color:#1a1a1a;">최근 날짜: <b>{last.name.strftime('%Y-%m-%d')}</b></span>
  &nbsp;&nbsp;|&nbsp;&nbsp;
  <span style="color:#e3b341;">%K = <b>{k_now:.1f}</b></span>
  &nbsp;&nbsp;
  <span style="color:#a5d6ff;">%D = <b>{d_now:.1f}</b></span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 렌더 헬퍼 — 지표 카드 + 차트 + 거래 내역 + 신호 테이블
# ─────────────────────────────────────────────────────────────────────────────
def highlight_row(row):
    color = "background-color:#e6f4ea;" if row["수익률(%)"] > 0 \
            else "background-color:#fce8e6;"
    return [color] * len(row)


def render_analysis(df_r, trades_r, ticker_r, label, hold_days_r, overbought_r, oversold_r):
    n2b = int(df_r["second_buy"].sum())
    n2s = int(df_r["second_sell"].sum())
    n1b = int(df_r["first_buy"].sum())
    n1s = int(df_r["first_sell"].sum())

    if not trades_r.empty:
        avg_ret  = trades_r["수익률(%)"].mean()
        win_rate = (trades_r["수익률(%)"] > 0).mean() * 100
    else:
        avg_ret = win_rate = 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("★ 2nd 매수 신호", f"{n2b}회", f"1st 무시: {n1b}회")
    m2.metric("★ 2nd 매도 신호", f"{n2s}회", f"1st 무시: {n1s}회")
    m3.metric("평균 수익률", f"{avg_ret:.2f}%", f"보유기간 {hold_days_r}일 기준")
    m4.metric("승률", f"{win_rate:.1f}%", f"총 {len(trades_r)}회 거래")

    st.markdown("<br>", unsafe_allow_html=True)

    fig = build_chart(df_r, f"{ticker_r} ({label})", overbought_r, oversold_r)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 백테스트 거래 내역 (두 번째 매수 신호 기준)")

    if trades_r.empty:
        st.info("해당 기간에 두 번째 매수 신호가 없습니다. 기간이나 파라미터를 조정해 보세요.")
    else:
        styled = (
            trades_r.style
            .apply(highlight_row, axis=1)
            .format({"매수가": "{:.2f}", "매도가": "{:.2f}", "수익률(%)": "{:+.2f}%"})
            .set_properties(**{"color": "#1a1a1a", "border": "1px solid #d0d0d0"})
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown("#### 📊 거래별 수익률")
        bar_colors = ["#3fb950" if r > 0 else "#f85149" for r in trades_r["수익률(%)"]]
        bar_fig = go.Figure(go.Bar(
            x=trades_r["매수일"], y=trades_r["수익률(%)"],
            marker_color=bar_colors,
            text=[f"{r:+.1f}%" for r in trades_r["수익률(%)"]],
            textposition="outside",
        ))
        bar_fig.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(color="#1a1a1a"), height=280,
            yaxis=dict(gridcolor="#e0e0e0", zeroline=True, zerolinecolor="#555555"),
            xaxis=dict(gridcolor="#e0e0e0"),
            showlegend=False, margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("---")
    col_buy, col_sell = st.columns(2)
    with col_buy:
        st.markdown("#### 🟢 두 번째 매수 신호 발생일")
        sb_df = df_r[df_r["second_buy"]][["Close", "%K", "%D"]].copy()
        sb_df.index = sb_df.index.strftime("%Y-%m-%d")
        sb_df.columns = ["종가", "%K", "%D"]
        st.dataframe(sb_df.round(2), use_container_width=True)
    with col_sell:
        st.markdown("#### 🔴 두 번째 매도 신호 발생일")
        ss_df = df_r[df_r["second_sell"]][["Close", "%K", "%D"]].copy()
        ss_df.index = ss_df.index.strftime("%Y-%m-%d")
        ss_df.columns = ["종가", "%K", "%D"]
        st.dataframe(ss_df.round(2), use_container_width=True)

    return avg_ret, win_rate, len(trades_r)


# ─────────────────────────────────────────────────────────────────────────────
# 비교 모드 분기
# ─────────────────────────────────────────────────────────────────────────────
if compare_mode and df_wk is not None:
    # ── 비교 요약 배너 ──────────────────────────────────────────────────────
    st.markdown("### ⚖️ 일봉 vs 주봉 비교 요약")

    def summary_stats(trades_r):
        if trades_r is None or trades_r.empty:
            return 0.0, 0.0, 0
        return (trades_r["수익률(%)"].mean(),
                (trades_r["수익률(%)"] > 0).mean() * 100,
                len(trades_r))

    ar_d, wr_d, cnt_d = summary_stats(trades)
    ar_w, wr_w, cnt_w = summary_stats(trades_wk)

    def delta_color(val):
        return "#3fb950" if val >= 0 else "#f85149"

    diff_ret  = ar_w  - ar_d
    diff_wr   = wr_w  - wr_d
    diff_cnt  = cnt_w - cnt_d

    st.markdown(f"""
    <div style="background:#f5f5f5;border:1px solid #d0d0d0;border-radius:10px;
                padding:16px 24px;margin-bottom:20px;">
      <table style="width:100%;border-collapse:collapse;font-family:monospace;">
        <tr style="color:#555555;font-size:0.8rem;">
          <th style="text-align:left;padding:4px 12px;">지표</th>
          <th style="text-align:center;padding:4px 12px;">일봉 (1d)</th>
          <th style="text-align:center;padding:4px 12px;">주봉 (1wk)</th>
          <th style="text-align:center;padding:4px 12px;">차이 (주봉-일봉)</th>
        </tr>
        <tr style="color:#1a1a1a;font-size:1rem;">
          <td style="padding:6px 12px;">평균 수익률</td>
          <td style="text-align:center;padding:6px 12px;">{ar_d:+.2f}%</td>
          <td style="text-align:center;padding:6px 12px;">{ar_w:+.2f}%</td>
          <td style="text-align:center;padding:6px 12px;color:{delta_color(diff_ret)};font-weight:700;">{diff_ret:+.2f}%</td>
        </tr>
        <tr style="color:#1a1a1a;font-size:1rem;">
          <td style="padding:6px 12px;">승률</td>
          <td style="text-align:center;padding:6px 12px;">{wr_d:.1f}%</td>
          <td style="text-align:center;padding:6px 12px;">{wr_w:.1f}%</td>
          <td style="text-align:center;padding:6px 12px;color:{delta_color(diff_wr)};font-weight:700;">{diff_wr:+.1f}%p</td>
        </tr>
        <tr style="color:#1a1a1a;font-size:1rem;">
          <td style="padding:6px 12px;">거래 횟수</td>
          <td style="text-align:center;padding:6px 12px;">{cnt_d}회</td>
          <td style="text-align:center;padding:6px 12px;">{cnt_w}회</td>
          <td style="text-align:center;padding:6px 12px;color:#555555;">{diff_cnt:+d}회</td>
        </tr>
      </table>
    </div>
    """, unsafe_allow_html=True)

    tab_d, tab_w = st.tabs(["📅 일봉 (1d)", "📆 주봉 (1wk)"])
    with tab_d:
        render_analysis(df, trades, ticker, "일봉 1d", hold_days, overbought, oversold)
    with tab_w:
        render_analysis(df_wk, trades_wk, ticker, "주봉 1wk", hold_days, overbought, oversold)

else:
    render_analysis(df, trades, ticker, "일봉 1d", hold_days, overbought, oversold)

# ─────────────────────────────────────────────────────────────────────────────
# Mag 7 신호 점검 & Discord 알림
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 Mag 7 전 종목 SSO 신호 점검")

wh_url = st.session_state.get("discord_url", "")


def _render_mag7_table(alerts: list) -> None:
    rows = []
    for a in alerts:
        sig = a["signal"]
        if sig == "second_buy":
            sig_label = "★ 2nd Buy 🟢"
        elif sig == "second_sell":
            sig_label = "★ 2nd Sell 🔴"
        else:
            sig_label = "─"
        rows.append({
            "종목": f"{a['name']} ({a['ticker']})",
            "날짜": a["date"],
            "종가": a["price"],
            "%K":   a["k_now"],
            "%D":   a["d_now"],
            "신호": sig_label,
        })

    result_df = pd.DataFrame(rows)

    def style_mag7(row):
        if "Buy"  in str(row["신호"]): return ["background-color:#e6f4ea"] * len(row)
        if "Sell" in str(row["신호"]): return ["background-color:#fce8e6"] * len(row)
        return ["background-color:#f5f5f5"] * len(row)

    st.dataframe(
        result_df.style
        .apply(style_mag7, axis=1)
        .set_properties(**{"color": "#1a1a1a", "border": "1px solid #d0d0d0"}),
        use_container_width=True,
        hide_index=True,
    )


def _send_mag7_discord(alerts: list, interval: str, wh_url: str) -> None:
    """신호 발생 종목만 골라 Discord로 전송. 중복 방지를 위해 sent_keys 추적."""
    signal_alerts = [a for a in alerts if a["signal"] is not None]
    if not signal_alerts:
        return

    sent_keys: set = st.session_state.setdefault("mag7_sent_keys", set())
    new_alerts = []
    for a in signal_alerts:
        key = f"{a['ticker']}_{a['signal']}_{a['date']}_{interval}"
        if key not in sent_keys:
            new_alerts.append(a)
            sent_keys.add(key)

    if not new_alerts:
        st.info("📭 이미 전송된 신호입니다 (중복 전송 방지)")
        return

    embeds = [build_embed(a["ticker"], a["signal"], a["row"], interval)
              for a in new_alerts]
    ok = False
    for i in range(0, len(embeds), 10):
        ok = send_discord(wh_url, embeds[i:i + 10])

    if ok:
        names = ", ".join(a["ticker"] for a in new_alerts)
        st.success(f"📨 Discord 전송 완료 — {names}")
    else:
        st.error("❌ Discord 전송 실패 — Webhook URL을 확인하세요.")


# ── 자동 스캔 (페이지 로드 시 항상 실행, 일봉+주봉 동시) ────────────────────
with st.spinner("Mag 7 종목 스캔 중 (일봉 + 주봉)..."):
    alerts_1d  = scan_mag7("1d",  k_period, k_smooth, d_smooth, overbought, oversold, _tickers=tuple(MAG7.keys()))
    alerts_1wk = scan_mag7("1wk", k_period, k_smooth, d_smooth, overbought, oversold, _tickers=tuple(MAG7.keys()))
    alerts = alerts_1d  # 현재 봉 신호 테이블은 선택된 타임프레임 기준 유지

# 버튼은 캐시 무효화(강제 새로고침) 용도
if mag7_btn:
    scan_mag7.clear()
    with st.spinner("Mag 7 새로고침 중..."):
        alerts_1d  = scan_mag7("1d",  k_period, k_smooth, d_smooth, overbought, oversold, _tickers=tuple(MAG7.keys()))
        alerts_1wk = scan_mag7("1wk", k_period, k_smooth, d_smooth, overbought, oversold, _tickers=tuple(MAG7.keys()))
        alerts = alerts_1d

# ── 현재 봉 신호 테이블 ───────────────────────────────────────────────────────
st.markdown("#### 📡 현재 봉 신호")
tab_cur_d, tab_cur_w = st.tabs(["📅 일봉 (1d)", "📆 주봉 (1wk)"])
with tab_cur_d:
    _render_mag7_table(alerts_1d)
    sig_d = [a for a in alerts_1d if a["signal"] is not None]
    if sig_d:
        st.success(f"✅ 일봉 신호 발생 종목: {len(sig_d)}개")
        if wh_url:
            _send_mag7_discord(alerts_1d, "1d", wh_url)
    else:
        st.info("일봉 기준 현재 봉 신호 없음")
with tab_cur_w:
    _render_mag7_table(alerts_1wk)
    sig_w = [a for a in alerts_1wk if a["signal"] is not None]
    if sig_w:
        st.success(f"✅ 주봉 신호 발생 종목: {len(sig_w)}개")
        if wh_url:
            _send_mag7_discord(alerts_1wk, "1wk", wh_url)
    else:
        st.info("주봉 기준 현재 봉 신호 없음")

# ── 최근 1개월 신호 섹션 (일봉 + 주봉 통합) ──────────────────────────────────
st.markdown("#### 📅 최근 1개월 내 2번째 신호 발생 종목 (일봉 + 주봉)")

recent_rows = []
for tf_label, tf_alerts in [("일봉", alerts_1d), ("주봉", alerts_1wk)]:
    for a in tf_alerts:
        for rs in a["recent_signals"]:
            sig_label = "★ 2nd Buy 🟢" if rs["signal"] == "second_buy" else "★ 2nd Sell 🔴"
            recent_rows.append({
                "TF":     tf_label,
                "종목":   f"{a['name']} ({a['ticker']})",
                "신호일": rs["date"],
                "종가":   rs["price"],
                "%K":     rs["k"],
                "%D":     rs["d"],
                "신호":   sig_label,
            })

if recent_rows:
    recent_df = pd.DataFrame(recent_rows)
    recent_df["_sig_order"] = recent_df["신호"].apply(lambda x: 0 if "Buy" in str(x) else 1)
    recent_df = recent_df.sort_values(["_sig_order", "신호일"], ascending=[True, False]).drop(columns=["_sig_order"])

    def style_recent(row):
        if "Buy"  in str(row["신호"]): return ["background-color:#e6f4ea"] * len(row)
        if "Sell" in str(row["신호"]): return ["background-color:#fce8e6"] * len(row)
        return ["background-color:#f5f5f5"] * len(row)

    st.dataframe(
        recent_df.style
        .apply(style_recent, axis=1)
        .set_properties(**{"color": "#1a1a1a", "border": "1px solid #d0d0d0"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"총 {len(recent_rows)}건 — 최근 30일 기준 (일봉+주봉)")
else:
    st.info("최근 1개월 내 2번째 신호 발생 종목 없음")

# ── auto_alert Discord 전송 ───────────────────────────────────────────────────
if auto_alert and wh_url:
    sig_cnt = sum(1 for a in alerts if a["signal"])
    if sig_cnt:
        st.warning(f"⚡ 자동 알림: Mag 7 신호 {sig_cnt}개 → Discord 전송")
        _send_mag7_discord(alerts, mag7_interval, wh_url)
    else:
        st.caption("⚡ 자동 알림 대기 중 — 현재 신호 없음")

# ── 일일 리포트 자동 전송 ──────────────────────────────────────────────────────
if daily_report_on and wh_url:
    result = check_and_send_daily_report(
        wh_url, mag7_interval,
        k_period, k_smooth, d_smooth,
        overbought, oversold,
        int(report_hour), int(report_minute),
    )
    if result == "sent":
        st.success(f"📨 일일 리포트 전송 완료 ({int(report_hour):02d}:{int(report_minute):02d})")
    elif result == "already_sent":
        st.caption(f"📋 오늘 일일 리포트 이미 전송됨 ({int(report_hour):02d}:{int(report_minute):02d} 예약)")
    elif result == "not_yet":
        st.caption(f"⏰ 일일 리포트 대기 중 — {int(report_hour):02d}:{int(report_minute):02d} 전송 예정")
    else:
        st.error("❌ 일일 리포트 전송 실패")

# ─────────────────────────────────────────────────────────────────────────────
# AI 딥다이브 분석
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 🤖 AI 딥다이브 분석")

_ai_cache_key = (str(ticker), int(k_period), int(k_smooth), int(d_smooth),
                 int(overbought), int(oversold), int(hold_days),
                 str(start_date), str(end_date))
_ai_force = st.button("🔄 AI 분석 새로고침", key="sso_ai_refresh")

if _ai_force or st.session_state.get("sso_ai_cache_key") != _ai_cache_key:
    st.session_state["sso_ai_cache_key"] = _ai_cache_key
    st.session_state.pop("sso_ai_result", None)  # 캐시 초기화

if "sso_ai_result" not in st.session_state:
    _do_ai = True
else:
    _do_ai = False

if _do_ai:
    try:
        _gemini_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=_gemini_key)
    except Exception as _e:
        st.error(f"❌ Gemini API 키 오류: {_e}")
        st.stop()

    # ── Mag 7 백테스트 랭킹 스캔 ──────────────────────────────────────────────
    _MAG7_SCAN = {
        "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA",
        "GOOGL": "Alphabet", "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla",
        "BTC-USD": "Bitcoin",
    }
    # 현재 분석 종목이 Mag7에 없으면 추가
    if ticker not in _MAG7_SCAN:
        _MAG7_SCAN[ticker] = ticker

    _scan_rows = []
    with st.spinner("📊 Mag 7 백테스트 스캔 중..."):
        for _t, _name in _MAG7_SCAN.items():
            try:
                _df_s, _tr_s = load_and_process(
                    _t, start_date, end_date, "1d",
                    k_period, k_smooth, d_smooth, overbought, oversold, hold_days,
                    total_capital, risk_per_trade, stop_loss, partial_exit_pct, target_profit_pct,
                )
                if _df_s is None or _tr_s is None or _tr_s.empty:
                    _scan_rows.append({
                        "종목": f"{_t} ({_name})", "거래횟수": 0,
                        "평균수익률(%)": 0.0, "승률(%)": 0.0,
                        "최고(%)": 0.0, "최저(%)": 0.0, "종합점수": 0.0,
                    })
                else:
                    _ar = _tr_s["수익률(%)"].mean()
                    _wr = (_tr_s["수익률(%)"] > 0).mean() * 100
                    _sc = _ar * (_wr / 100)  # 평균수익률 × 승률 → 종합점수
                    _scan_rows.append({
                        "종목": f"{_t} ({_name})",
                        "거래횟수": len(_tr_s),
                        "평균수익률(%)": round(_ar, 2),
                        "승률(%)": round(_wr, 1),
                        "최고(%)": round(_tr_s["수익률(%)"].max(), 2),
                        "최저(%)": round(_tr_s["수익률(%)"].min(), 2),
                        "종합점수": round(_sc, 2),
                    })
            except Exception:
                pass

    _scan_df = pd.DataFrame(_scan_rows).sort_values("종합점수", ascending=False).reset_index(drop=True)
    _scan_df.index = _scan_df.index + 1  # 1위부터 표시

    st.markdown("### 🏆 Mag 7 백테스트 랭킹 (현재 파라미터 기준)")
    st.caption(f"파라미터: k_period={k_period}, k_smooth={k_smooth}, d_smooth={d_smooth}  |  보유기간={hold_days}일  |  기간: {start_date} ~ {end_date}")

    def _style_rank(row):
        if row.name == 1:
            return ["background-color:#fff8e1; font-weight:700"] * len(row)
        if row.name == 2:
            return ["background-color:#f5f5f5; font-weight:600"] * len(row)
        if row.name == 3:
            return ["background-color:#fafafa"] * len(row)
        return [""] * len(row)

    st.dataframe(
        _scan_df.style.apply(_style_rank, axis=1)
        .format({"평균수익률(%)": "{:+.2f}%", "승률(%)": "{:.1f}%",
                 "최고(%)": "{:+.2f}%", "최저(%)": "{:+.2f}%", "종합점수": "{:.2f}"}),
        use_container_width=True,
    )

    # 랭킹 요약 문자열 (프롬프트용)
    _rank_lines = []
    for _i, _rr in _scan_df.iterrows():
        _rank_lines.append(
            f"  {_i}위 {_rr['종목']}: 평균수익률={_rr['평균수익률(%)']:+.2f}%  "
            f"승률={_rr['승률(%)']}%  거래{_rr['거래횟수']}회  종합점수={_rr['종합점수']:.2f}"
        )
    _best_ticker = _scan_df.iloc[0]["종목"] if not _scan_df.empty else "N/A"

    # ── 현재 종목 SSO 데이터 요약 ─────────────────────────────────────────────
    _last = df.dropna(subset=["%K", "%D"]).iloc[-1]
    _k_now, _d_now = _last["%K"], _last["%D"]
    _price_now = _last["Close"]
    _date_now  = _last.name.strftime("%Y-%m-%d")

    # 최근 30일 신호 수집
    _cutoff = pd.Timestamp.today() - pd.Timedelta(days=30)
    _recent_sigs = []
    for _dt, _row in df[df.index >= _cutoff].dropna(subset=["%K", "%D"]).iterrows():
        if _row.get("second_buy"):
            _recent_sigs.append(f"  {str(_dt)[:10]} 2nd Buy  (종가:{_row['Close']:.2f}  %K:{_row['%K']:.1f})")
        elif _row.get("second_sell"):
            _recent_sigs.append(f"  {str(_dt)[:10]} 2nd Sell (종가:{_row['Close']:.2f}  %K:{_row['%K']:.1f})")

    # 백테스트 성과 요약
    if not trades.empty:
        _avg_ret  = trades["수익률(%)"].mean()
        _win_rate = (trades["수익률(%)"] > 0).mean() * 100
        _n_trades = len(trades)
        _best     = trades["수익률(%)"].max()
        _worst    = trades["수익률(%)"].min()
    else:
        _avg_ret = _win_rate = _n_trades = _best = _worst = 0

    # 주봉 데이터 (있는 경우)
    _wk_info = ""
    if df_wk is not None:
        _last_wk = df_wk.dropna(subset=["%K", "%D"]).iloc[-1]
        _wk_zone = "과매수" if _last_wk["%K"] > overbought else ("과매도" if _last_wk["%K"] < oversold else "중립")
        _wk_info = f"  주봉 %K={_last_wk['%K']:.1f}  %D={_last_wk['%D']:.1f}  구간:{_wk_zone}"

    _zone_str = "과매수" if _k_now > overbought else ("과매도" if _k_now < oversold else "중립")

    _depth = "딥다이브"
    _max_tokens = 16384

    # ── 전 종목 현재 SSO 상태 요약 (alerts_1d / alerts_1wk 활용) ─────────────
    def _zone(k):
        if k is None: return "N/A"
        if k > overbought: return "과매수"
        if k < oversold:   return "과매도"
        return "중립"

    def _sig_label(sig):
        if sig == "second_buy":  return "2nd Buy 🟢"
        if sig == "second_sell": return "2nd Sell 🔴"
        return "신호없음"

    _all_stocks_lines = []
    _all_stocks_lines.append("  [일봉]")
    for _a in alerts_1d:
        _k = _a.get("k_now"); _d = _a.get("d_now"); _p = _a.get("price")
        _all_stocks_lines.append(
            f"  {_a['name']}({_a['ticker']}): "
            f"종가={_p if _p else 'N/A'}  %K={_k if _k else 'N/A'}  %D={_d if _d else 'N/A'}  "
            f"구간={_zone(_k)}  신호={_sig_label(_a.get('signal'))}"
        )
    _all_stocks_lines.append("  [주봉]")
    for _a in alerts_1wk:
        _k = _a.get("k_now"); _d = _a.get("d_now"); _p = _a.get("price")
        _all_stocks_lines.append(
            f"  {_a['name']}({_a['ticker']}): "
            f"종가={_p if _p else 'N/A'}  %K={_k if _k else 'N/A'}  %D={_d if _d else 'N/A'}  "
            f"구간={_zone(_k)}  신호={_sig_label(_a.get('signal'))}"
        )

    # 최근 1개월 2nd 신호 발생 종목 요약
    _all_recent_lines = []
    for _tf_label, _tf_alerts in [("일봉", alerts_1d), ("주봉", alerts_1wk)]:
        for _a in _tf_alerts:
            for _rs in _a.get("recent_signals", []):
                _sl = "2nd Buy 🟢" if _rs["signal"] == "second_buy" else "2nd Sell 🔴"
                _all_recent_lines.append(
                    f"  [{_tf_label}] {_a['name']}({_a['ticker']}) "
                    f"{_rs['date']}  종가={_rs['price']}  %K={_rs['k']}  → {_sl}"
                )

    _strategy_block = """
### 6. 구체적 매매 전략 제언
- 현재 포지션 여부에 따른 행동 지침 (진입 / 관망 / 청산)
- 추천 매수 진입 가격대 및 손절 / 목표가
- 단기(1-2주) vs 중기(1개월) 전략 차이
"""

    _risk_block = """
### 7. 리스크 시나리오 분석
- Bull Case / Base Case / Bear Case (각 확률 포함)
- 핵심 위험 트리거와 대응 방법
"""

    _prompt = f"""
당신은 기술적 분석 전문가입니다. 아래 SSO(슬로우 스토캐스틱 오실레이터) 데이터를 바탕으로 **한국어로** 심층 분석을 제공하세요.

===== 분석 대상 종목 =====
종목: {ticker}
분석 기준일: {_date_now}
현재 종가: {_price_now:.2f} USD

===== 현재 SSO 상태 (일봉) =====
%K = {_k_now:.2f}  |  %D = {_d_now:.2f}
구간: {_zone_str}  (과매수 기준: {overbought} / 과매도 기준: {oversold})
파라미터: %K기간={k_period}, K스무딩={k_smooth}, D스무딩={d_smooth}
{_wk_info}

===== 최근 30일 2번째 신호 =====
{chr(10).join(_recent_sigs) if _recent_sigs else "  최근 30일 내 2번째 신호 없음"}

===== {ticker} 백테스트 성과 ({start_date} ~ {end_date}, 보유기간 {hold_days}일) =====
총 거래 횟수: {_n_trades}회
평균 수익률: {_avg_ret:.2f}%
승률: {_win_rate:.1f}%
최고 거래: {_best:.2f}%  |  최저 거래: {_worst:.2f}%

===== 전 종목 현재 SSO 상태 (Mag7 + BTC) =====
{chr(10).join(_all_stocks_lines)}

===== 최근 1개월 2번째 신호 발생 (전 종목) =====
{chr(10).join(_all_recent_lines) if _all_recent_lines else "  최근 1개월 신호 없음"}

===== 전 종목 백테스트 랭킹 (동일 파라미터) =====
종합점수 = 평균수익률 × (승률/100) — 높을수록 이 전략과 궁합이 좋음
{chr(10).join(_rank_lines)}
→ 현재 파라미터 기준 최적 종목: {_best_ticker}

===== 분석 요청 (딥다이브 모드) =====

### 1. 분석 대상 종목 ({ticker}) 현재 상태 진단
- %K/%D 위치와 추세 방향 해석
- 현재 구간(과매수/과매도/중립)의 의미와 다음 예상 전개

### 2. {ticker} 2번째 신호 분석
- 최근 신호의 신뢰도 평가
- 일봉/주봉 신호 일치 여부 및 의미

### 3. {ticker} 백테스트 성과 해석
- 현재 파라미터(k={k_period}, {k_smooth}, {d_smooth})가 {ticker}에 적합한지
- 승률·수익률을 기준으로 전략의 강점과 약점

### 4. 전 종목 SSO 시장 종합 진단
- 현재 일봉/주봉 기준 과매수·과매도·중립 종목 분포 분석
- 시장 전체적으로 어느 사이클 국면인지 (상승 초입 / 상승 과열 / 하락 초입 / 바닥권)
- 일봉과 주봉 신호가 일치하는 종목 vs 엇갈리는 종목 해석

### 5. 최근 2번째 신호 발생 종목 심층 분석
- 최근 1개월 내 신호 발생 종목들의 공통 패턴과 의미
- 2nd Buy 신호 종목 중 지금 진입 우선순위 및 이유
- 2nd Sell 신호 종목의 하락 위험 수준 평가

### 6. SSO 전략 최적 종목 분석
- 백테스트 랭킹 기준 이 전략에 가장 잘 맞는 종목과 이유
- 1~3위 종목의 공통점 (변동성, 추세 특성 등)
- 현재 분석 종목({ticker})의 랭킹 위치가 갖는 의미

### 7. 포트폴리오 관점 종합 결론
- 전 종목 SSO 상태를 고려한 포트폴리오 배분 제언
- 지금 당장 매수/매도/관망해야 할 종목 각각 명시
- 가장 주목해야 할 핵심 포인트 3가지
{_strategy_block}
{_risk_block}

**분석은 간결하고 실용적으로, 투자자가 즉시 참고할 수 있는 수준으로 작성하세요.**
본 분석은 투자 권유가 아니며 참고 목적입니다.
"""

    with st.spinner("🤖 Gemini AI 분석 중..."):
        try:
            _model = genai.GenerativeModel("gemini-2.5-flash")
            _safety = [
                {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            _resp = _model.generate_content(
                _prompt,
                generation_config={"max_output_tokens": _max_tokens, "temperature": 0.7},
                safety_settings=_safety,
            )
            _analysis_text = _resp.text
            st.session_state["sso_ai_result"] = {
                "text": _analysis_text,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "ticker": ticker,
                "scan_df": _scan_df,
            }
            _ai_cache_save("sso_ai_result", st.session_state["sso_ai_result"])
        except Exception as _e:
            st.error(f"❌ Gemini 분석 실패: {_e}")
            st.stop()

# ── 결과 표시 (캐시 또는 방금 생성) ─────────────────────────────────────────
if "sso_ai_result" in st.session_state:
    _res = st.session_state["sso_ai_result"]
    _analysis_text = _res["text"]

    if not _do_ai:  # 캐시에서 불러온 경우 랭킹 테이블 재표시
        _scan_df = _res.get("scan_df")
        if _scan_df is not None:
            st.markdown("### 🏆 Mag 7 백테스트 랭킹 (현재 파라미터 기준)")
            st.caption(f"파라미터: k_period={k_period}, k_smooth={k_smooth}, d_smooth={d_smooth}  |  보유기간={hold_days}일  |  기간: {start_date} ~ {end_date}")

            def _style_rank(row):
                if row.name == 1: return ["background-color:#fff8e1; font-weight:700"] * len(row)
                if row.name == 2: return ["background-color:#f5f5f5; font-weight:600"] * len(row)
                if row.name == 3: return ["background-color:#fafafa"] * len(row)
                return [""] * len(row)

            st.dataframe(
                _scan_df.style.apply(_style_rank, axis=1)
                .format({"평균수익률(%)": "{:+.2f}%", "승률(%)": "{:.1f}%",
                         "최고(%)": "{:+.2f}%", "최저(%)": "{:+.2f}%", "종합점수": "{:.2f}"}),
                use_container_width=True,
            )

    st.markdown(f"### 🤖 AI 분석 결과 — {_res['ticker']}")
    st.caption(f"생성 시각: {_res['time']}  |  딥다이브 모드")
    st.markdown(f"""
<div style="background:#f9f9f9;border:1px solid #d0d0d0;border-radius:12px;
            padding:20px 24px;margin:12px 0;font-size:0.95rem;line-height:1.7;color:#1a1a1a;">
{_analysis_text.replace(chr(10), '<br>')}
</div>
""", unsafe_allow_html=True)

    st.download_button(
        "📥 분석 결과 다운로드 (.txt)",
        data=f"AI 딥다이브 분석 — {_res['ticker']} ({_res['time']})\n\n{_analysis_text}",
        file_name=f"sso_ai_{_res['ticker']}_{_res['time'].replace(':', '').replace(' ', '_')}.txt",
        mime="text/plain",
    )

# ─────────────────────────────────────────────────────────────────────────────
# 하단 설명
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📖 전략 원리 요약 보기"):
    st.markdown("""
    ### 슬로우 스토캐스틱 오실레이터 (SSO) — 9, 5, 3 전략

    | 항목 | 내용 |
    |------|------|
    | **권장 파라미터** | k_period=**9**, k_smooth=**5**, d_smooth=**3** |
    | **9기간의 이유** | 단기 사이클의 절반 길이 → 사이클 파동이 지표에 선명히 나타남 |
    | **첫 번째 신호** | 과매수/과매도 구간 최초 교차 → **노이즈, 무시** |
    | **두 번째 신호** | 재진입 후 두 번째 교차 → **단기+장기 사이클 동기화 순간** |
    | **성공 원리** | 합산 원리(Summation) + 사이클 동기화(Synchronicity) |

    > **핵심 규칙**: 과매수·과매도 구간의 **첫 번째 교차는 무시**, **두 번째 교차에서만 매매**한다.
    """)