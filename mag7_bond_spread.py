"""MAG7 회사채 수익률과 미국 10년물 스프레드.

회사채 수익률은 FINRA가 공개하는 TRACE 일별 최종 체결 수익률
(`endOfDayPriceYield`)이다. API 키가 필요 없다. 미국 10년물은 이 앱의
다른 금리와 같이 FRED DGS10의 실제 고시값만 쓴다.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from fredapi import Fred

from dashboard_ui import dashboard_plotly_chart

TRACE_URL = (
    "https://services-dynarep.ddwa.finra.org/public/reporting/v2/data/group/"
    "FixedIncomeMarket/name"
)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)
TARGET_MATURITY = date(2035, 7, 1)
PAGE_LIMIT = 5000
YIELD_MIN = 0.0
YIELD_MAX = 25.0

SECURITY_FIELDS = [
    "cusip",
    "issueSymbolIdentifier",
    "issuerName",
    "couponRate",
    "maturityDate",
    "isCallable",
    "isConvertible",
    "is144A",
    "productSubTypeCode",
    "couponType",
    "lastSaleYield",
    "lastSalePrice",
    "lastTradeDate",
    "moodysRating",
    "standardAndPoorsRating",
]
HISTORY_FIELDS = [
    "tradeDate",
    "cusip",
    "issueSymbolIdentifier",
    "lastSaleYield",
    "lastSalePrice",
]

# 검색어만으로는 자회사나 비슷한 상호가 섞인다. 응답 뒤 발행사 이름으로 가른다.
ISSUERS = [
    {
        "ticker": "AAPL",
        "name": "Apple",
        "searches": [["APPLE"]],
        "allowed_issuers": {"APPLE INC"},
    },
    {
        "ticker": "MSFT",
        "name": "Microsoft",
        "searches": [["MICROSOFT"]],
        "allowed_issuers": {"MICROSOFT CORP", "MICROSOFT CORPORATION"},
    },
    {
        "ticker": "GOOGL",
        "name": "Alphabet",
        "searches": [["ALPHABET"], ["GOOGLE"]],
        "allowed_issuers": {"ALPHABET INC", "GOOGLE LLC", "GOOGLE INC"},
    },
    {
        "ticker": "AMZN",
        "name": "Amazon",
        "searches": [["AMAZON"]],
        "allowed_issuers": {"AMAZON COM INC", "AMAZON.COM INC", "AMAZON.COM, INC."},
    },
    {
        "ticker": "NVDA",
        "name": "NVIDIA",
        "searches": [["NVIDIA"]],
        "allowed_issuers": {"NVIDIA CORP", "NVIDIA CORPORATION"},
    },
    {
        "ticker": "META",
        "name": "Meta",
        "searches": [["META", "PLATFORMS"]],
        "allowed_issuers": {"META PLATFORMS INC"},
    },
    {
        "ticker": "TSLA",
        "name": "Tesla",
        "searches": [["TESLA"]],
        "allowed_issuers": {"TESLA INC"},
    },
]

COLORS = {
    "AAPL": "#e0b15a",
    "MSFT": "#3d8bfd",
    "GOOGL": "#2e9d57",
    "AMZN": "#e07a00",
    "NVDA": "#7cb342",
    "META": "#7e57c2",
    "TSLA": "#d45d5d",
}


class TraceUnavailable(Exception):
    """FINRA TRACE 공개 데이터를 읽지 못했을 때."""


def chart_window(start_date: str, today: date | None = None) -> tuple[str, str]:
    """사이드바 기간을 따르되, 1년보다 짧으면 최소 1년을 요청한다."""
    today = today or date.today()
    requested = datetime.strptime(start_date, "%Y-%m-%d").date()
    floor = today - timedelta(days=365)
    return min(requested, floor).isoformat(), today.isoformat()


def format_coupon(value) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    if "." not in text:
        return f"{text}.00"
    whole, fraction = text.split(".")
    if len(fraction) < 2:
        fraction = fraction.ljust(2, "0")
    return f"{whole}.{fraction}"


def _issuer_name(row: dict) -> str:
    return (row.get("issuerName") or "").strip().upper()


def rejection_reasons(row: dict, allowed_issuers: set[str], today: date) -> list[str]:
    """선정 조건에 걸리면 이유를 돌려준다. 비어 있으면 후보이다."""
    reasons = []
    if _issuer_name(row) not in allowed_issuers:
        reasons.append("모회사 발행이 아님")
    if row.get("productSubTypeCode") != "CORP":
        reasons.append("회사채(CORP)가 아님")
    if row.get("isConvertible") != "N":
        reasons.append("전환사채")
    if row.get("is144A") != "N":
        reasons.append("144A 사모")
    if row.get("couponType") != "FXPV":
        reasons.append("고정 표면금리(FXPV)가 아님")
    maturity_text = row.get("maturityDate")
    if row.get("couponRate") is None or not maturity_text:
        reasons.append("쿠폰 또는 만기 없음")
        return reasons
    try:
        maturity = date.fromisoformat(maturity_text)
    except ValueError:
        reasons.append("만기 날짜를 읽을 수 없음")
        return reasons
    if maturity < today:
        reasons.append("만기 경과")
    return reasons


def _distance(row: dict) -> int:
    return abs((date.fromisoformat(row["maturityDate"]) - TARGET_MATURITY).days)


def _trade_ordinal(value) -> int:
    try:
        return date.fromisoformat(value).toordinal()
    except (TypeError, ValueError):
        return 0


def choose_issue(rows: list[dict], allowed_issuers: set[str], today: date) -> dict:
    """후보 중 2035-07-01에 만기가 가장 가까운 채권과 제외 내역을 고른다."""
    eligible = []
    rejected = []
    for row in rows:
        reasons = rejection_reasons(row, allowed_issuers, today)
        if reasons:
            rejected.append({**row, "reasons": reasons})
        else:
            eligible.append(row)
    eligible.sort(
        key=lambda row: (
            _distance(row),
            -_trade_ordinal(row.get("lastTradeDate")),
            row.get("cusip") or "",
        )
    )
    return {
        "selected": eligible[0] if eligible else None,
        "runner_up": eligible[1] if len(eligible) > 1 else None,
        "rejected": rejected,
    }


def _bond_label(row: dict) -> str:
    coupon = format_coupon(row["couponRate"]) if row.get("couponRate") is not None else "?"
    return (
        f"{row.get('issuerName') or ''} {coupon}% {row.get('maturityDate') or ''}"
        f" (CUSIP {row.get('cusip') or '없음'}, FINRA {row.get('issueSymbolIdentifier') or '없음'})"
    )


def _missing_note(name: str, rejected: list[dict]) -> str:
    if not rejected:
        return (
            f"{name}: FINRA TRACE 회사·기관채에서 이 회사 이름의 채권을 찾지 못했습니다. "
            "수익률을 만들지 않습니다."
        )
    lines = [
        f"**{name}:** 2035년에 가까운 미상환 고정금리 공모 회사채가 없습니다. "
        "아래는 TRACE에서 조회된 채권과 제외 이유이며, 값을 채우지 않았습니다."
    ]
    shown = sorted(rejected, key=lambda row: row.get("maturityDate") or "")
    for row in shown[:8]:
        lines.append(f"· {_bond_label(row)} — {', '.join(row['reasons'])}")
    hidden = len(shown) - 8
    if hidden > 0:
        lines.append(f"· 그 밖 {hidden}건도 같은 기준으로 제외했습니다.")
    return "\n\n".join(lines)


def _selection_note(bond: dict, first_trade: str | None, chart_start: str) -> str:
    notes = []
    maturity = date.fromisoformat(bond["maturity"])
    if maturity.year != 2035:
        notes.append(
            f"2035년 만기 채권이 없어, 2035-07-01에 가장 가까운 만기인 {bond['maturity']} 채권을 사용합니다."
        )
    if first_trade and first_trade > chart_start:
        notes.append(f"TRACE 체결 수익률은 {first_trade}부터 있습니다.")
    return " ".join(notes)


def _trace_headers(token: str) -> dict:
    return {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://www.finra.org",
        "Referer": "https://www.finra.org/finra-data/fixed-income/corp-and-agency/trade",
        "X-XSRF-TOKEN": token,
    }


def _open_trace() -> tuple[requests.Session, str]:
    session = requests.Session()
    try:
        session.get(
            f"{TRACE_URL}/CorporateAndAgencySecurities",
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
            timeout=60,
        )
    except requests.RequestException as exc:
        raise TraceUnavailable(f"FINRA TRACE에 연결하지 못했습니다 ({exc.__class__.__name__}).") from exc
    token = session.cookies.get("XSRF-TOKEN")
    if not token:
        raise TraceUnavailable("FINRA TRACE가 접속 토큰을 내려주지 않았습니다.")
    return session, token


def _post_page(session: requests.Session, token: str, dataset: str, payload: dict, offset: int) -> tuple[list, int]:
    last_error = "알 수 없는 응답"
    for attempt in range(3):
        try:
            response = session.post(
                f"{TRACE_URL}/{dataset}",
                json={**payload, "limit": PAGE_LIMIT, "offset": offset},
                headers=_trace_headers(token),
                timeout=120,
            )
        except requests.RequestException as exc:
            last_error = exc.__class__.__name__
            time.sleep(0.5 * 2**attempt)
            continue
        if response.status_code in (502, 503, 524) and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.status_code != 200:
            detail = response.text[:180].replace("\n", " ")
            raise TraceUnavailable(f"FINRA TRACE {dataset} 응답 {response.status_code}: {detail}")
        try:
            body = response.json()
        except ValueError as exc:
            raise TraceUnavailable(f"FINRA TRACE {dataset} 응답이 JSON이 아닙니다.") from exc
        if body.get("status") != "success":
            message = body.get("statusMessage") or body.get("status")
            raise TraceUnavailable(f"FINRA TRACE {dataset} 요청이 거절되었습니다: {message}")
        returned = body.get("returnBody") or {}
        data = returned.get("data")
        if isinstance(data, str) and data.strip():
            rows = json.loads(data)
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        totals = (returned.get("headers") or {}).get("Record-Total") or [len(rows)]
        return rows, int(totals[0] or 0)
    raise TraceUnavailable(f"FINRA TRACE {dataset} 요청에 실패했습니다 ({last_error}).")


def _query(session: requests.Session, token: str, dataset: str, payload: dict) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    previous_first = None
    total = None
    while total is None or offset < total:
        page, total = _post_page(session, token, dataset, payload, offset)
        if not page:
            break
        first = json.dumps(page[0], sort_keys=True, default=str)
        if previous_first == first:
            break
        previous_first = first
        rows.extend(page)
        offset += len(page)
        if offset > PAGE_LIMIT * 20:
            break
    return rows


def _search_issuer(session: requests.Session, token: str, words: list[str]) -> list[dict]:
    payload = {
        "fields": SECURITY_FIELDS,
        "multiFieldMatchFilters": [
            {
                "searchValue": word,
                "fuzzy": False,
                "synonym": False,
                "fields": [{"name": "issuerName", "boost": 1}],
            }
            for word in words
        ],
        "sortFields": ["+maturityDate", "+cusip"],
    }
    return _query(session, token, "corporateAndAgencySecurities", payload)


def _history(session: requests.Session, token: str, cusips: list[str], start: str, end: str) -> list[dict]:
    if not cusips:
        return []
    payload = {
        "fields": HISTORY_FIELDS,
        "domainFilters": [{"fieldName": "cusip", "values": cusips}],
        "dateRangeFilters": [{"fieldName": "tradeDate", "startDate": start, "endDate": end}],
        "sortFields": ["+tradeDate", "+cusip"],
    }
    return _query(session, token, "endOfDayPriceYield", payload)


def _clean_yields(rows: list[dict]) -> tuple[list[dict], int]:
    cleaned = []
    dropped = 0
    for row in rows:
        raw = row.get("lastSaleYield")
        if raw is None or raw == "":
            continue
        try:
            yield_pct = float(raw)
        except (TypeError, ValueError):
            dropped += 1
            continue
        if not (YIELD_MIN < yield_pct < YIELD_MAX):
            dropped += 1
            continue
        price = row.get("lastSalePrice")
        try:
            price = float(price) if price not in (None, "") else None
        except (TypeError, ValueError):
            price = None
        trade_date = row.get("tradeDate")
        if not trade_date or not row.get("cusip"):
            continue
        cleaned.append(
            {
                "cusip": row["cusip"],
                "date": trade_date,
                "yield_pct": yield_pct,
                "price": price,
            }
        )
    return cleaned, dropped


def fetch_mag7_panel(chart_start: str, chart_end: str, today: date | None = None) -> dict:
    """TRACE에서 채권을 고르고, 기간 안 체결 수익률만 가져온다."""
    today = today or date.fromisoformat(chart_end)
    session, token = _open_trace()
    bonds = []
    errors = []
    for spec in ISSUERS:
        try:
            found: dict[str, dict] = {}
            for words in spec["searches"]:
                for row in _search_issuer(session, token, words):
                    cusip = row.get("cusip")
                    if cusip:
                        found[cusip] = row
            choice = choose_issue(list(found.values()), spec["allowed_issuers"], today)
        except TraceUnavailable as exc:
            errors.append(f"{spec['ticker']}: {exc}")
            bonds.append(
                {
                    "ticker": spec["ticker"],
                    "name": spec["name"],
                    "status": "error",
                    "note": f"{spec['name']} 채권 목록을 가져오지 못했습니다. {exc}",
                }
            )
            continue
        selected = choice["selected"]
        if selected is None:
            bonds.append(
                {
                    "ticker": spec["ticker"],
                    "name": spec["name"],
                    "status": "missing",
                    "note": _missing_note(spec["name"], choice["rejected"]),
                }
            )
            continue
        bonds.append(
            {
                "ticker": spec["ticker"],
                "name": spec["name"],
                "status": "selected",
                "issuer": selected.get("issuerName"),
                "coupon": float(selected["couponRate"]),
                "maturity": selected["maturityDate"],
                "cusip": selected["cusip"],
                "symbol": selected.get("issueSymbolIdentifier"),
                "callable": selected.get("isCallable"),
                "moodys": selected.get("moodysRating"),
                "sp": selected.get("standardAndPoorsRating"),
                "last_trade": selected.get("lastTradeDate"),
                "distance_days": _distance(selected),
                "runner_up": _bond_label(choice["runner_up"]) if choice["runner_up"] else "",
                "note": "",
            }
        )

    cusips = [bond["cusip"] for bond in bonds if bond.get("status") == "selected"]
    try:
        history_rows = _history(session, token, cusips, chart_start, chart_end) if cusips else []
    except TraceUnavailable as exc:
        errors.append(f"체결 이력: {exc}")
        history_rows = []
    yields, dropped = _clean_yields(history_rows)
    first_by_cusip: dict[str, str] = {}
    for row in yields:
        current = first_by_cusip.get(row["cusip"])
        if current is None or row["date"] < current:
            first_by_cusip[row["cusip"]] = row["date"]
    for bond in bonds:
        if bond.get("status") == "selected":
            bond["note"] = _selection_note(bond, first_by_cusip.get(bond["cusip"]), chart_start)
    return {
        "chart_start": chart_start,
        "chart_end": chart_end,
        "bonds": bonds,
        "yields": yields,
        "dropped_yields": dropped,
        "errors": errors,
    }


def load_dgs10_prints(start_date: str, api_key: str) -> pd.Series:
    """FRED DGS10 고시값. 결측은 앞 값으로 채우지 않는다."""
    series = Fred(api_key=api_key).get_series("DGS10", observation_start=start_date)
    values = pd.to_numeric(series, errors="coerce").dropna()
    values.index = pd.to_datetime(values.index).tz_localize(None).normalize()
    return values[~values.index.duplicated(keep="last")].sort_index()


def compute_spreads(yield_rows: list[dict], dgs10: pd.Series) -> pd.DataFrame:
    """같은 날짜에 회사채 체결과 DGS10이 모두 있는 날만 스프레드를 계산한다."""
    columns = ["cusip", "date", "bond_yield", "ust10y", "spread_bp"]
    if not yield_rows or dgs10 is None or len(dgs10) == 0:
        return pd.DataFrame(columns=columns)
    treasury = pd.to_numeric(dgs10, errors="coerce").dropna()
    treasury.index = pd.to_datetime(treasury.index).tz_localize(None).normalize()
    treasury = treasury[~treasury.index.duplicated(keep="last")].sort_index()
    treasury_frame = treasury.rename("ust10y").rename_axis("date").reset_index()
    bonds = pd.DataFrame(yield_rows)
    bonds["date"] = pd.to_datetime(bonds["date"]).dt.tz_localize(None).dt.normalize()
    bonds["bond_yield"] = pd.to_numeric(bonds["yield_pct"], errors="coerce")
    bonds = bonds.dropna(subset=["cusip", "date", "bond_yield"])
    bonds = bonds.sort_values(["cusip", "date"]).drop_duplicates(["cusip", "date"], keep="last")
    merged = bonds.merge(treasury_frame, on="date", how="inner")
    merged["spread_bp"] = (merged["bond_yield"] - merged["ust10y"]) * 100.0
    return merged.loc[:, columns].sort_values(["cusip", "date"]).reset_index(drop=True)


def lookback_change(spread: pd.Series, days: int) -> dict:
    """최근 관측과, 그로부터 days일 이전의 마지막 실제 관측을 비교한다."""
    series = pd.to_numeric(spread, errors="coerce").dropna().sort_index()
    series = series[~series.index.duplicated(keep="last")]
    latest_date = series.index[-1]
    latest = float(series.iloc[-1])
    prior = series.loc[: latest_date - pd.Timedelta(days=days)]
    if prior.empty:
        return {
            "latest": latest,
            "latest_date": latest_date,
            "prior": None,
            "prior_date": None,
            "change": None,
        }
    prior_value = float(prior.iloc[-1])
    return {
        "latest": latest,
        "latest_date": latest_date,
        "prior": prior_value,
        "prior_date": prior.index[-1],
        "change": latest - prior_value,
    }


@st.cache_data(ttl=3600, show_spinner=False)
def cached_mag7_panel(chart_start: str, chart_end: str) -> dict:
    return fetch_mag7_panel(chart_start, chart_end)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_dgs10(start_date: str) -> pd.Series:
    api_key = st.secrets["FRED_API_KEY"]
    return load_dgs10_prints(start_date, api_key)


def _fmt_date(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return "—"
    return stamp.strftime("%Y-%m-%d")


def _fmt_change(change, prior_date) -> str:
    if change is None:
        return "비교할 이전 체결 없음"
    return f"{change:+.1f} bp ({_fmt_date(prior_date)})"


def _spread_figure(spreads: pd.DataFrame, bonds: list[dict]) -> go.Figure:
    fig = go.Figure()
    by_cusip = {bond["cusip"]: bond for bond in bonds if bond.get("status") == "selected"}
    for cusip, bond in by_cusip.items():
        group = spreads[spreads["cusip"] == cusip]
        if group.empty:
            continue
        name = f"{bond['ticker']} {format_coupon(bond['coupon'])}% {bond['maturity']}"
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["spread_bp"],
                name=name,
                mode="lines+markers",
                connectgaps=False,
                line=dict(color=COLORS.get(bond["ticker"], "#d8ad55"), width=2),
                marker=dict(size=5),
                customdata=list(zip(group["bond_yield"], group["ust10y"])),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "%{x|%Y-%m-%d}<br>"
                    "스프레드 %{y:.1f} bp<br>"
                    "회사채 %{customdata[0]:.2f}%<br>"
                    "미국 10년 %{customdata[1]:.2f}%"
                    "<extra></extra>"
                ),
            )
        )
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        height=560,
        title_text="<b>MAG7 회사채 수익률 − 미국 10년물</b>",
        hovermode="x unified",
        yaxis_title="스프레드 (bp)",
        xaxis_title="체결일",
    )
    return fig


def _identity_table(bonds: list[dict], spreads: pd.DataFrame) -> pd.DataFrame:
    records = []
    for bond in bonds:
        if bond.get("status") != "selected":
            continue
        group = spreads[spreads["cusip"] == bond["cusip"]] if not spreads.empty else spreads
        if group.empty:
            span = "이 기간의 체결 수익률 없음"
        else:
            span = f"{_fmt_date(group['date'].min())} – {_fmt_date(group['date'].max())} ({len(group)}일)"
        records.append(
            {
                "티커": bond["ticker"],
                "발행사": bond["issuer"],
                "표면금리": f"{format_coupon(bond['coupon'])}%",
                "만기": bond["maturity"],
                "CUSIP": bond["cusip"],
                "FINRA 심볼": bond.get("symbol") or "—",
                "무디스": bond.get("moodys") or "—",
                "S&P": bond.get("sp") or "—",
                "콜": "가능" if bond.get("callable") == "Y" else ("없음" if bond.get("callable") == "N" else "—"),
                "2035-07-01까지": f"{bond['distance_days']}일",
                "체결 구간": span,
                "비고": bond.get("note") or "",
            }
        )
    return pd.DataFrame(records)


def _render_body(start_date: str, dgs10: pd.Series | None) -> None:
    chart_start, chart_end = chart_window(start_date)
    with st.spinner("📡 FINRA TRACE에서 MAG7 회사채 수익률을 가져오는 중..."):
        panel = cached_mag7_panel(chart_start, chart_end)
        treasury_error = None
        if dgs10 is None:
            try:
                treasury = cached_dgs10(chart_start)
            except Exception as exc:
                treasury = None
                treasury_error = exc
        else:
            treasury = dgs10

    for message in panel.get("errors") or []:
        st.warning(message)

    spreads = compute_spreads(panel["yields"], treasury) if treasury is not None else compute_spreads([], pd.Series(dtype=float))
    selected = [bond for bond in panel["bonds"] if bond.get("status") == "selected"]
    if selected and not spreads.empty:
        metric_rows = [selected[:3], selected[3:]]
        for group in metric_rows:
            if not group:
                continue
            columns = st.columns(3)
            for column, bond in zip(columns, group):
                series = spreads.loc[spreads["cusip"] == bond["cusip"]].set_index("date")["spread_bp"]
                with column:
                    if series.empty:
                        st.metric(bond["ticker"], "—", help=bond.get("note") or "이 기간에 겹치는 체결이 없습니다.")
                        st.caption(bond.get("note") or "이 기간에 DGS10과 겹치는 체결이 없습니다.")
                        continue
                    week = lookback_change(series, 7)
                    month = lookback_change(series, 30)
                    latest_row = spreads.loc[spreads["cusip"] == bond["cusip"]].iloc[-1]
                    delta = None if week["change"] is None else f"{week['change']:+.1f} bp"
                    st.metric(
                        f"{bond['ticker']} · {_fmt_date(week['latest_date'])}",
                        f"{week['latest']:.1f} bp",
                        delta=delta,
                        delta_color="inverse",
                        help=(
                            f"{bond['issuer']} {format_coupon(bond['coupon'])}% {bond['maturity']}\n"
                            f"CUSIP {bond['cusip']} · FINRA {bond.get('symbol') or '—'}\n"
                            "스프레드가 커지면 신용 프리미엄이 넓어진 것입니다.\n"
                            f"1주: {_fmt_change(week['change'], week['prior_date'])}\n"
                            f"1개월: {_fmt_change(month['change'], month['prior_date'])}"
                        ),
                    )
                    st.caption(
                        f"1주 {_fmt_change(week['change'], week['prior_date'])} · "
                        f"1개월 {_fmt_change(month['change'], month['prior_date'])}\n\n"
                        f"회사채 {latest_row['bond_yield']:.2f}% − 미국 10년 {latest_row['ust10y']:.2f}%"
                    )

    if treasury_error is not None:
        st.warning(f"미국 10년물(DGS10)을 가져오지 못해 스프레드를 계산하지 않았습니다. {treasury_error}")
    elif treasury is not None and selected and spreads.empty:
        st.warning("회사채 체결과 미국 10년물 고시일이 겹치는 날이 없어 스프레드를 그리지 않았습니다.")

    if not spreads.empty:
        dashboard_plotly_chart(_spread_figure(spreads, panel["bonds"]), use_container_width=True)
        st.download_button(
            "MAG7 스프레드 CSV",
            _download_csv(spreads, selected),
            file_name=f"mag7_ust_spread_{chart_end}.csv",
            mime="text/csv",
            key="mag7_spread_csv",
        )

    table = _identity_table(panel["bonds"], spreads)
    if not table.empty:
        st.dataframe(table, use_container_width=True, hide_index=True)

    for bond in panel["bonds"]:
        if bond.get("status") == "missing":
            st.info(bond["note"])
        elif bond.get("status") == "error":
            st.warning(bond["note"])

    runners = [bond for bond in selected if bond.get("runner_up")]
    if runners:
        with st.expander("같은 조건에서 다음으로 가까웠던 채권"):
            for bond in runners:
                st.markdown(f"- **{bond['ticker']}** {bond['runner_up']}")

    if panel.get("dropped_yields"):
        st.caption(
            f"0% 이하이거나 25% 이상인 TRACE 수익률 {panel['dropped_yields']}건은 "
            "체결 오류로 보고 차트에서 빼 두었습니다."
        )
    st.caption(
        f"요청 구간 {chart_start} – {chart_end}. "
        "등급은 TRACE에 실린 값입니다. 콜 가능 채권의 체결 수익률은 콜 조건을 반영할 수 있어 CDS와 같지는 않습니다."
    )


def _download_csv(spreads: pd.DataFrame, bonds: list[dict]) -> str:
    meta = pd.DataFrame(
        [
            {
                "cusip": bond["cusip"],
                "ticker": bond["ticker"],
                "issuer": bond["issuer"],
                "coupon": bond["coupon"],
                "maturity": bond["maturity"],
                "symbol": bond.get("symbol"),
            }
            for bond in bonds
        ]
    )
    frame = spreads.merge(meta, on="cusip", how="left")
    frame["date"] = frame["date"].map(_fmt_date)
    ordered = frame[
        ["date", "ticker", "issuer", "coupon", "maturity", "cusip", "symbol", "bond_yield", "ust10y", "spread_bp"]
    ]
    return ordered.to_csv(index=False)


def _selected_ai_line(bond: dict, spreads: pd.DataFrame) -> str:
    """한 회사의 최신 스프레드와 1주·1개월 변화를 문장으로 만든다."""
    label = (
        f"{bond['ticker']} {bond.get('issuer') or bond.get('name') or ''} "
        f"{format_coupon(bond['coupon'])}% {bond['maturity']}"
        f" (CUSIP {bond.get('cusip') or '없음'}, FINRA {bond.get('symbol') or '없음'})"
    )
    group = spreads[spreads["cusip"] == bond.get("cusip")] if not spreads.empty else spreads
    if group.empty:
        extra = f" {bond['note']}" if bond.get("note") else ""
        return f"- {label}: 같은 날의 체결과 DGS10이 없어 스프레드를 계산하지 못했습니다.{extra}"
    ordered = group.sort_values("date")
    latest_row = ordered.iloc[-1]
    series = ordered.set_index("date")["spread_bp"]
    week = lookback_change(series, 7)
    month = lookback_change(series, 30)
    text = (
        f"- {label}: 최신 스프레드 {float(latest_row['spread_bp']):.1f} bp "
        f"({_fmt_date(latest_row['date'])}, 회사채 {float(latest_row['bond_yield']):.2f}%, "
        f"같은 날 DGS10 {float(latest_row['ust10y']):.2f}%). "
        f"약 1주 전 대비 {_fmt_change(week['change'], week['prior_date'])}. "
        f"약 1개월 전 대비 {_fmt_change(month['change'], month['prior_date'])}."
    )
    if bond.get("note"):
        text += f" {bond['note']}"
    return text


def build_mag7_ai_context(
    panel: dict,
    treasury: pd.Series | None = None,
    treasury_error: str | None = None,
) -> str:
    """Gemini에 넘길 MAG7 스프레드 문맥. 입력만 사용하고 네트워크를 호출하지 않는다."""
    lines = [
        "MAG7 회사채 당일 마지막 체결 수익률에서 같은 날 미국 10년물(FRED DGS10)을 뺀 스프레드입니다.",
        "만기가 같은 국채와의 G-스프레드가 아니라 10년물 단일 금리와의 차이입니다.",
        "단위는 bp(1%p = 100bp)이고, 아래 증감은 스프레드 bp의 차이입니다. 퍼센트 변화율이 아닙니다.",
        "회사채 체결과 DGS10이 같은 날 모두 있는 경우만 계산했고, 빈 날은 보간하지 않았습니다.",
        "수치가 없으면 값을 추정하거나 채우지 마세요.",
    ]
    bonds = list(panel.get("bonds") or [])
    if treasury_error or treasury is None or len(treasury) == 0:
        lines.append(
            "미국 10년물(DGS10) 최신 고시를 불러오지 못했습니다. "
            "회사채와 10년물의 스프레드는 계산할 수 없습니다."
            + (f" ({treasury_error})" if treasury_error else "")
        )
        spreads = pd.DataFrame(columns=["cusip", "date", "bond_yield", "ust10y", "spread_bp"])
    else:
        clean = pd.to_numeric(treasury, errors="coerce").dropna().sort_index()
        if clean.empty:
            lines.append("미국 10년물(DGS10) 고시값에 숫자가 없어 스프레드를 계산할 수 없습니다.")
            spreads = pd.DataFrame(columns=["cusip", "date", "bond_yield", "ust10y", "spread_bp"])
        else:
            lines.append(
                f"미국 10년물(DGS10) 최근 고시: {float(clean.iloc[-1]):.2f}% ({_fmt_date(clean.index[-1])})."
            )
            spreads = compute_spreads(panel.get("yields") or [], clean)

    if not bonds:
        lines.append(
            "MAG7 회사채 − 미국 10년물 스프레드 데이터는 현재 이용할 수 없습니다. "
            "회사별 채권과 스프레드를 받지 못했습니다."
        )
    for bond in bonds:
        status = bond.get("status")
        ticker = bond.get("ticker") or ""
        if status == "selected":
            lines.append(_selected_ai_line(bond, spreads))
        elif status == "missing":
            note = bond.get("note") or f"{bond.get('name') or ticker}: 조건에 맞는 채권이 없습니다."
            lines.append(
                f"- {ticker}: 적합한 회사채가 없습니다. 스프레드를 계산하지 않았고 값을 만들지 않았습니다.\n{note}"
            )
        elif status == "error":
            lines.append(
                f"- {ticker}: 데이터를 불러오지 못했습니다. "
                f"{bond.get('note') or '이 회사의 스프레드는 현재 이용할 수 없습니다.'}"
            )
        else:
            lines.append(f"- {ticker}: 스프레드 데이터를 현재 이용할 수 없습니다.")

    errors = [str(item) for item in (panel.get("errors") or []) if item]
    if errors:
        lines.append("조회 중 확인된 오류: " + "; ".join(errors))
    if panel.get("chart_start") and panel.get("chart_end"):
        lines.append(f"요청 구간: {panel['chart_start']} – {panel['chart_end']}.")
    return "\n".join(lines)


def mag7_ai_context(start_date: str) -> str:
    """차트와 같은 캐시로 MAG7 스프레드 문맥을 만든다. 실패해도 예외를 올리지 않는다."""
    try:
        chart_start, chart_end = chart_window(start_date)
        panel = cached_mag7_panel(chart_start, chart_end)
    except Exception as exc:
        return (
            "MAG7 회사채 − 미국 10년물 스프레드 데이터는 현재 이용할 수 없습니다. "
            "FINRA TRACE에서 회사채 수익률을 가져오지 못했습니다. "
            f"({exc.__class__.__name__}: {exc})"
        )
    treasury = None
    treasury_error = None
    try:
        treasury = cached_dgs10(chart_start)
    except Exception as exc:
        treasury_error = f"{exc.__class__.__name__}: {exc}"
    try:
        return build_mag7_ai_context(panel, treasury, treasury_error)
    except Exception as exc:
        return (
            "MAG7 회사채 − 미국 10년물 스프레드 데이터는 현재 이용할 수 없습니다. "
            f"스프레드를 정리하지 못했습니다. ({exc.__class__.__name__}: {exc})"
        )


def render_mag7_credit_spreads(start_date: str, dgs10: pd.Series | None = None) -> None:
    """Macro 페이지에 MAG7 신용 스프레드 섹션을 그린다. 실패해도 예외를 올리지 않는다."""
    st.markdown("### 📉 MAG7 회사채 − 미국 10년물 스프레드")
    st.caption(
        "각 회사의 TRACE 당일 마지막 체결 수익률에서, 같은 날 고시된 미국 10년물(FRED DGS10)을 뺀 값입니다. "
        "만기가 같은 국채와의 G-스프레드가 아니라 10년물 단일 금리와의 차이입니다. "
        "단위는 베이시스 포인트(1%p = 100bp)이고, 숫자가 클수록 국채 대비 신용 프리미엄이 넓습니다. "
        "두 값이 모두 있는 날만 표시하며 빈 날은 보간하지 않습니다. 카드의 증감은 1주 전 마지막 체결 대비입니다."
    )
    st.caption(
        "채권은 모회사 명의의 회사채(CORP) 가운데 전환사채가 아니고, 144A 사모가 아니며, "
        "고정 표면금리(FXPV)이고, 만기가 남아 있는 종목 중 만기일이 2035-07-01에 가장 가까운 것을 고릅니다. "
        "TRACE 공개 항목에는 선·후순위 코드가 없어 그 코드로 걸러 내지는 못합니다. "
        "데이터는 FINRA TRACE 일별 최종 체결 수익률(무료, API 키 없음, 비상업 공개)과 FRED DGS10입니다. "
        "사이드바 기간이 1년보다 짧으면 이 섹션은 최소 1년을 요청하고, 채권 발행이 그보다 늦으면 첫 체결일부터 보여 줍니다."
    )
    try:
        _render_body(start_date, dgs10)
    except Exception as exc:
        st.warning(
            "MAG7 회사채 스프레드를 불러오지 못했습니다. 다른 매크로 지표는 그대로 사용할 수 있습니다. "
            f"({exc.__class__.__name__}: {exc})"
        )
