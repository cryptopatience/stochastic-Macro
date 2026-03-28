"""
==============================================================================
AI 통합 딥다이브 분석 — SSO 딥다이브 + 매크로 딥다이브 자동 통합
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

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
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# 새로고침 후 복원
for _k in ["sso_ai_result", "macro_ai_result", "unified_ai_result"]:
    if _k not in st.session_state:
        _c = _ai_cache_load(_k)
        if _c:
            st.session_state[_k] = _c

import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta
import google.generativeai as genai
import requests
import os as _os

# ─────────────────────────────────────────────────────────────────────────────
# 라이트 테마
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #ffffff; color: #1a1a1a; }
    section[data-testid="stSidebar"] { background-color: #f5f5f5; }
    h1, h2, h3, h4 { color: #1a1a1a !important; }
    hr { border-color: #d0d0d0; }
    .analysis-box {
        background: #f9f9f9;
        border: 1px solid #d0d0d0;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 12px 0;
        font-size: 0.95rem;
        line-height: 1.7;
        color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 AI 통합 딥다이브 분석")
st.markdown("**SSO 기술적 분석 + 매크로·신용위험 분석**을 통합해 Gemini AI가 최종 투자 판단을 제공합니다.")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# API 설정
# ─────────────────────────────────────────────────────────────────────────────
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    FRED_API_KEY   = st.secrets["FRED_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    fred = Fred(api_key=FRED_API_KEY)
except Exception as e:
    st.error(f"❌ API 키 오류: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 데이터 수집 함수 (session_state 결과가 없을 때 자동 실행용)
# ─────────────────────────────────────────────────────────────────────────────
MAG7 = {
    "AAPL": "🍎 Apple", "MSFT": "🪟 Microsoft", "NVDA": "💚 NVIDIA",
    "GOOGL": "🔍 Alphabet", "AMZN": "📦 Amazon",
    "META": "👁️ Meta", "TSLA": "⚡ Tesla", "BTC-USD": "₿ Bitcoin",
}

@st.cache_data(show_spinner=False, ttl=300)
def _get_sso_snapshot():
    """전 종목 현재 SSO 상태 수집 (일봉 + 주봉)"""
    end   = datetime.today()
    start = end - timedelta(days=120)
    k_period, k_smooth, d_smooth, ob, os_ = 9, 5, 3, 80, 20
    lines = {"일봉": [], "주봉": [], "recent": []}

    for ticker, name in MAG7.items():
        for interval, label in [("1d", "일봉"), ("1wk", "주봉")]:
            try:
                df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                                 end=end.strftime("%Y-%m-%d"),
                                 interval=interval, auto_adjust=True, progress=False)
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(df) < k_period + k_smooth + d_smooth:
                    continue

                low_min  = df["Low"].rolling(k_period).min()
                high_max = df["High"].rolling(k_period).max()
                kf = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-10)
                df["%K"] = kf.rolling(k_smooth).mean()
                df["%D"] = df["%K"].rolling(d_smooth).mean()

                K, D = df["%K"].values, df["%D"].values
                n = len(df)
                sb = np.zeros(n, bool); ss = np.zeros(n, bool)
                bd = sd = False
                for i in range(1, n):
                    if np.isnan(K[i]) or np.isnan(D[i]): continue
                    if K[i-1] < D[i-1] and K[i] >= D[i] and K[i] < os_ + 10:
                        if not bd: bd = True
                        else: sb[i] = True; bd = False
                    if K[i] > os_ + 15 and D[i] > os_ + 15: bd = False
                    if K[i-1] > D[i-1] and K[i] <= D[i] and K[i] > ob - 10:
                        if not sd: sd = True
                        else: ss[i] = True; sd = False
                    if K[i] < ob - 15 and D[i] < ob - 15: sd = False
                df["sb"] = sb; df["ss"] = ss

                last = df.dropna(subset=["%K", "%D"]).iloc[-1]
                zone = "과매수" if last["%K"] > ob else ("과매도" if last["%K"] < os_ else "중립")
                sig  = "2nd Buy 🟢" if last["sb"] else ("2nd Sell 🔴" if last["ss"] else "신호없음")
                lines[label].append(
                    f"  {name}({ticker}): 종가={last['Close']:.2f}  %K={last['%K']:.1f}"
                    f"  %D={last['%D']:.1f}  구간={zone}  신호={sig}"
                )

                cutoff = end - timedelta(days=30)
                for dt, row in df[df.index >= pd.Timestamp(cutoff)].dropna(subset=["%K","%D"]).iterrows():
                    if row["sb"]:
                        lines["recent"].append(f"  [{label}] {name}({ticker}) {str(dt)[:10]} → 2nd Buy  %K={row['%K']:.1f}")
                    elif row["ss"]:
                        lines["recent"].append(f"  [{label}] {name}({ticker}) {str(dt)[:10]} → 2nd Sell %K={row['%K']:.1f}")
            except Exception:
                continue
    return lines


@st.cache_data(show_spinner=False, ttl=3600)
def _get_macro_snapshot():
    """FRED 주요 매크로 지표 수집"""
    start = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")
    series = {
        "DGS10":         "10년물 국채금리",
        "DGS2":          "2년물 국채금리",
        "T10Y2Y":        "장단기 금리차(10Y-2Y)",
        "FEDFUNDS":      "연준 기준금리",
        "BAMLH0A0HYM2":  "하이일드 스프레드",
        "BAMLC0A0CM":    "투자등급 스프레드",
        "DRCCLACBS":     "신용카드 연체율",
        "DROCLACBS":     "오토론 연체율",
        "DRCRELEXFACBS": "CRE 연체율",
    }
    data = {}
    for sid, name in series.items():
        try:
            s = fred.get_series(sid, observation_start=start)
            s = s.sort_index().ffill().dropna()
            if len(s): data[name] = round(float(s.iloc[-1]), 3)
        except Exception:
            pass
    return data


def _run_gemini(prompt: str, max_tokens: int = 16384) -> str:
    model = genai.GenerativeModel("gemini-2.5-flash")
    safety = [
        {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    resp = model.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": 0.7},
        safety_settings=safety,
    )
    return resp.text


# ─────────────────────────────────────────────────────────────────────────────
# 새로고침 버튼
# ─────────────────────────────────────────────────────────────────────────────
_force = st.button("🔄 전체 분석 새로고침", key="unified_refresh")
if _force:
    for k in ["sso_ai_result", "macro_ai_result", "unified_ai_result",
              "unified_cache_key", "p3_sso_done", "p3_macro_done"]:
        st.session_state.pop(k, None)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: SSO 딥다이브 자동 생성 (없으면)
# ─────────────────────────────────────────────────────────────────────────────
if "sso_ai_result" not in st.session_state:
    with st.spinner("📈 SSO 전 종목 스캔 + AI 딥다이브 분석 중..."):
        try:
            sso_snap = _get_sso_snapshot()
            sso_prompt = f"""
당신은 기술적 분석 전문가입니다. SSO(슬로우 스토캐스틱 오실레이터) 데이터를 바탕으로 **한국어로** 딥다이브 분석을 제공하세요.

===== 전 종목 SSO 현재 상태 =====
[일봉]
{chr(10).join(sso_snap['일봉']) if sso_snap['일봉'] else '  데이터 없음'}

[주봉]
{chr(10).join(sso_snap['주봉']) if sso_snap['주봉'] else '  데이터 없음'}

===== 최근 1개월 2번째 신호 발생 =====
{chr(10).join(sso_snap['recent']) if sso_snap['recent'] else '  최근 1개월 신호 없음'}

===== 딥다이브 분석 요청 =====

### 1. 전 종목 SSO 시장 종합 진단
- 현재 일봉/주봉 기준 과매수·과매도·중립 종목 분포
- 시장 전체 사이클 국면 (상승 초입 / 상승 과열 / 하락 초입 / 바닥권)

### 2. 2번째 신호 발생 종목 심층 분석
- 최근 1개월 신호 발생 종목의 공통 패턴
- 2nd Buy 종목 중 진입 우선순위 및 이유
- 2nd Sell 종목의 하락 위험 수준

### 3. 종목별 SSO 전략 판단
- Mag7 + BTC 각 종목: 매수 / 관망 / 매도 의견 및 근거

### 4. SSO 전략 최적 종목
- 현재 SSO 신호 기준 가장 매력적인 종목 Top 3

### 5. 종합 결론 및 핵심 포인트 3가지

**분석은 간결하고 실용적으로 작성하세요. 본 분석은 참고 목적입니다.**
"""
            sso_text = _run_gemini(sso_prompt)
            st.session_state["sso_ai_result"] = {
                "text": sso_text,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "ticker": "Mag7+BTC",
            }
            _ai_cache_save("sso_ai_result", st.session_state["sso_ai_result"])
        except Exception as e:
            st.error(f"❌ SSO 자동 분석 실패: {e}")
            st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: 매크로 딥다이브 자동 생성 (없으면)
# ─────────────────────────────────────────────────────────────────────────────
if "macro_ai_result" not in st.session_state:
    with st.spinner("🏦 FRED 매크로 데이터 수집 + AI 딥다이브 분석 중..."):
        try:
            macro_snap = _get_macro_snapshot()
            macro_lines = "\n".join(f"  {k}: {v}" for k, v in macro_snap.items())
            macro_prompt = f"""
당신은 거시경제·신용 리스크 전문가입니다. **한국어로** 딥다이브 분석을 제공하세요.

===== FRED 매크로 지표 =====
{macro_lines}

===== 딥다이브 분석 요청 =====

### 1. 거시경제 환경 심층 분석
- Fed 정책 사이클 위치 (긴축/완화/전환점)
- 수익률 곡선의 역사적 맥락과 의미

### 2. 신용 시장 리스크 평가
- 하이일드·투자등급 스프레드 현 수준의 역사적 위치
- 연체율 트렌드와 소비자·기업 스트레스 수준

### 3. 다중 시나리오 분석 (확률 포함)
- Bull Case / Base Case / Bear Case

### 4. 섹터별 리스크 평가
- 금융, 부동산, 소비재, 기술/성장주

### 5. 투자 전략 제언
- 채권·주식·대안자산 배분
- 현금 비중 및 리스크 관리

### 6. 핵심 모니터링 지표 및 트리거 레벨

**수치와 근거를 명확히 제시하세요. 본 분석은 참고 목적입니다.**
"""
            macro_text = _run_gemini(macro_prompt)
            st.session_state["macro_ai_result"] = {
                "text": macro_text,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            _ai_cache_save("macro_ai_result", st.session_state["macro_ai_result"])
        except Exception as e:
            st.error(f"❌ 매크로 자동 분석 실패: {e}")
            st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 사전 분석 완료 상태 표시
# ─────────────────────────────────────────────────────────────────────────────
sso_res   = st.session_state["sso_ai_result"]
macro_res = st.session_state["macro_ai_result"]

col_s, col_m = st.columns(2)
with col_s:
    st.success(f"✅ SSO 딥다이브 완료 ({sso_res['time']})")
with col_m:
    st.success(f"✅ 매크로 딥다이브 완료 ({macro_res['time']})")

# 원문 보기
with st.expander("📈 SSO 딥다이브 분석 원문 보기", expanded=False):
    st.markdown(sso_res["text"])
with st.expander("🏦 매크로 딥다이브 분석 원문 보기", expanded=False):
    st.markdown(macro_res["text"])

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: 통합 딥다이브 자동 생성
# ─────────────────────────────────────────────────────────────────────────────
_cache_key = (sso_res["time"], macro_res["time"])
if st.session_state.get("unified_cache_key") != _cache_key:
    st.session_state["unified_cache_key"] = _cache_key
    st.session_state.pop("unified_ai_result", None)

if "unified_ai_result" not in st.session_state:
    with st.spinner("🤖 Gemini AI 통합 딥다이브 분석 중... (잠시 기다려 주세요)"):
        try:
            unified_prompt = f"""
당신은 거시경제, 신용 리스크, 기술적 분석을 통합하는 최고 수준의 투자 전략가입니다.
아래 두 전문 AI 보고서를 종합하여 **한국어로** 최종 통합 투자 판단을 작성하세요.

════════════════════════════════════════════════════════════════
[보고서 A] SSO 기술적 딥다이브 분석
════════════════════════════════════════════════════════════════
{sso_res['text']}

════════════════════════════════════════════════════════════════
[보고서 B] 매크로·신용위험 딥다이브 분석
════════════════════════════════════════════════════════════════
{macro_res['text']}

════════════════════════════════════════════════════════════════
[최종 통합 분석 요청]
════════════════════════════════════════════════════════════════

### 1. 두 분석의 일치점 vs 상충점
- SSO 기술적 신호와 매크로 환경이 같은 방향을 가리키는 부분
- 서로 엇갈리거나 모순되는 부분과 그 의미

### 2. 시장 사이클 종합 판단
- 기술적 사이클(SSO)과 신용 사이클(매크로)의 현재 위치
- 두 사이클의 동기화 여부 및 디커플링 의미

### 3. 종목별 최종 투자 판단
- Mag 7 + BTC 각 종목: **매수 / 관망 / 매도** 명확히 제시
- 판단 근거: SSO 신호 + 매크로 환경 교차 분석
- 우선순위 Top 3 선정 및 이유

### 4. 포트폴리오 전략 제언
- 현재 매크로 환경에서 적정 주식 비중
- 섹터 배분 및 현금 비중 권고
- 단기(2주) / 중기(1~3개월) 전략 차이

### 5. 핵심 리스크 & 트리거
- 지금 가장 주시해야 할 리스크 Top 3
- 시나리오별 대응:
  - Upside: 기술적 + 매크로 동시 호전
  - Baseline: 현 상태 지속
  - Downside: 기술적 악화 + 매크로 악화

### 6. 지금 당장 해야 할 액션 3가지
- 투자자가 오늘 즉시 실행할 수 있는 구체적 행동 지침

**두 보고서의 핵심 인사이트를 녹여내되, 단순 요약이 아닌 통합적 시각으로 새로운 결론을 도출하세요.**
본 분석은 투자 권유가 아니며 참고 목적입니다.
"""
            unified_text = _run_gemini(unified_prompt, max_tokens=16384)
            st.session_state["unified_ai_result"] = {
                "text": unified_text,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            _ai_cache_save("unified_ai_result", st.session_state["unified_ai_result"])
        except Exception as e:
            st.error(f"❌ 통합 분석 실패: {e}")
            st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 결과 표시
# ─────────────────────────────────────────────────────────────────────────────
res = st.session_state["unified_ai_result"]

st.markdown("## 🤖 AI 통합 딥다이브 분석 결과")
st.caption(f"생성 시각: {res['time']}  |  딥다이브 모드  |  SSO + 매크로 통합")

st.markdown(f"""
<div class="analysis-box">
{res['text'].replace(chr(10), '<br>')}
</div>
""", unsafe_allow_html=True)

st.download_button(
    "📥 통합 분석 결과 다운로드 (.txt)",
    data=f"AI 통합 딥다이브 분석 ({res['time']})\n\n{res['text']}",
    file_name=f"unified_deepdive_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
    mime="text/plain",
)

# ─────────────────────────────────────────────────────────────────────────────
# Discord 일일 자동 발송 (매일 08:00)
# ─────────────────────────────────────────────────────────────────────────────
_WH_URL        = "https://discord.com/api/webhooks/1487415854839894076/i2HkxX91ZbcWFzOHe9QZjLvNNXPl-j6t1rZs2hnQcvC0gbzk0l0Ohyce2nXU5C3IYD0A"
_REPORT_HOUR   = 8
_REPORT_MINUTE = 0


def _truncate(text: str, limit: int = 3900) -> str:
    """Discord embed description 한도 내로 자르고 말줄임 표시."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n… (전체 내용은 대시보드에서 확인하세요)"


def _post_embeds(embeds: list) -> tuple[bool, str]:
    """embed 목록을 10개씩 나눠 전송. (성공여부, 오류메시지) 반환."""
    try:
        for i in range(0, len(embeds), 10):
            r = requests.post(_WH_URL, json={"embeds": embeds[i:i+10]}, timeout=20)
            if r.status_code not in (200, 204):
                return False, f"HTTP {r.status_code}: {r.text[:200]}"
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_and_send_ai_daily_report() -> tuple[str, str]:
    """매일 08:00 이후 최초 1회 AI 딥다이브 리포트를 Discord로 전송.
    반환: (상태코드, 오류메시지)"""
    now       = datetime.now()
    today_str = now.strftime("%Y-%m-%d")

    # session_state로 중복 방지 (Streamlit Cloud 호환)
    if st.session_state.get("ai_report_sent_date") == today_str:
        return "already_sent", ""

    if (now.hour, now.minute) < (_REPORT_HOUR, _REPORT_MINUTE):
        return "not_yet", ""

    sso_r   = st.session_state.get("sso_ai_result")
    macro_r = st.session_state.get("macro_ai_result")
    uni_r   = st.session_state.get("unified_ai_result")

    if not all([sso_r, macro_r, uni_r]):
        return "not_ready", ""

    today_label = f"{today_str} {_REPORT_HOUR:02d}:{_REPORT_MINUTE:02d}"

    embeds = [
        {
            "title":       f"📈 [1/3] SSO 딥다이브 분석 — {today_label}",
            "color":       0x3FB950,
            "description": _truncate(sso_r["text"]),
            "footer":      {"text": f"생성: {sso_r['time']}"},
        },
        {
            "title":       f"🏦 [2/3] 매크로 딥다이브 분석 — {today_label}",
            "color":       0x58A6FF,
            "description": _truncate(macro_r["text"]),
            "footer":      {"text": f"생성: {macro_r['time']}"},
        },
        {
            "title":       f"🤖 [3/3] AI 통합 딥다이브 분석 — {today_label}",
            "color":       0xE3B341,
            "description": _truncate(uni_r["text"]),
            "footer":      {"text": f"생성: {uni_r['time']}  |  AI 통합 대시보드"},
        },
    ]

    ok, err = _post_embeds(embeds)
    if ok:
        st.session_state["ai_report_sent_date"] = today_str
        return "sent", ""
    return "error", err


# ── 상태 표시 ─────────────────────────────────────────────────────────────────
st.markdown("---")
_result, _err = _check_and_send_ai_daily_report()

if _result == "sent":
    st.success(f"📨 AI 딥다이브 일일 리포트 Discord 전송 완료 (3건) — {_REPORT_HOUR:02d}:{_REPORT_MINUTE:02d}")
elif _result == "already_sent":
    st.caption(f"📋 오늘 AI 딥다이브 리포트 이미 전송됨 — {_REPORT_HOUR:02d}:{_REPORT_MINUTE:02d} 예약")
elif _result == "not_yet":
    st.caption(f"⏰ AI 딥다이브 일일 리포트 대기 중 — {_REPORT_HOUR:02d}:{_REPORT_MINUTE:02d} 전송 예정")
elif _result == "not_ready":
    st.caption("⏳ 분석 결과 생성 후 자동 전송됩니다")
elif _result == "error":
    st.error(f"❌ Discord 전송 실패: {_err}")
else:
    st.error("❌ Discord 전송 실패")
