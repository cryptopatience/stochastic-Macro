"""Restore login across refreshes without storing passwords in the browser."""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
import secrets
import threading
import time

import streamlit as st
import streamlit.components.v1 as components

TOKEN_TTL_SECONDS = 7 * 86400
_COMPONENT_DIR = Path(__file__).resolve().parent / "auth_component"
_storage = components.declare_component(
    "scgt_login_storage", path=str(_COMPONENT_DIR)
)


class LoginTokens:
    """Server-side session tokens keyed to the current password hash."""

    def __init__(self):
        self.tokens = {}
        self.lock = threading.Lock()

    def issue(self, password):
        with self.lock:
            self._purge_unlocked()
            token = secrets.token_urlsafe(32)
            self.tokens[token] = (
                _password_digest(password),
                time.time() + TOKEN_TTL_SECONDS,
            )
            return token

    def valid(self, token, password):
        with self.lock:
            record = self.tokens.get(token) if isinstance(token, str) else None
            return bool(
                record
                and record[1] > time.time()
                and hmac.compare_digest(record[0], _password_digest(password))
            )

    def revoke(self, token):
        with self.lock:
            if isinstance(token, str):
                self.tokens.pop(token, None)

    def _purge_unlocked(self):
        now = time.time()
        self.tokens = {key: value for key, value in self.tokens.items() if value[1] > now}


def _password_digest(password):
    return hashlib.sha256(password.encode("utf-8")).digest()


def passwords_match(entered, expected):
    return hmac.compare_digest(_password_digest(entered), _password_digest(expected))


def read_app_password():
    """Return APP_PASSWORD from Streamlit secrets or the environment."""
    value = None
    try:
        value = st.secrets.get("APP_PASSWORD")
    except Exception:
        value = None
    if value is None or str(value).strip() == "":
        value = os.environ.get("APP_PASSWORD")
    if value is None or str(value).strip() == "":
        return None
    return str(value)


@st.cache_resource
def _tokens():
    return LoginTokens()


def _request(action, token=None):
    st.session_state["_login_storage_request"] = {
        "action": action,
        "token": token,
        "request_id": secrets.token_hex(16),
    }


def _mount_storage(request):
    st.markdown(
        """
        <style>
        iframe[title="scgt_login_storage"] {
            position: absolute !important;
            width: 0 !important;
            height: 0 !important;
            border: 0 !important;
            visibility: hidden !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    result = _storage(**request, key="login_storage", default=None)
    if not result or result.get("request_id") != request["request_id"]:
        return None
    return result


def _show_login_form(password, tokens):
    st.markdown("## 🔒 통합 투자 대시보드")
    st.markdown("접속하려면 비밀번호를 입력하세요.")
    with st.form("login_form"):
        entered = st.text_input("비밀번호", type="password")
        submitted = st.form_submit_button("로그인", type="primary")
        if submitted:
            if passwords_match(entered, password):
                issued = tokens.issue(password)
                st.session_state["_login_token"] = issued
                st.session_state["authenticated"] = True
                _request("save", issued)
                st.rerun()
            st.error("비밀번호가 틀렸습니다.")


def _render_logout(tokens, token):
    if st.sidebar.button("🔓 로그아웃", key="logout_btn"):
        tokens.revoke(token)
        st.session_state.clear()
        _request("clear")
        st.rerun()


def require_login():
    """Block the page until a valid dashboard session exists."""
    password = read_app_password()
    if password is None:
        st.error(
            "이 대시보드는 비밀번호가 없으면 시작할 수 없습니다. "
            "운영자는 `.streamlit/secrets.toml` 또는 환경 변수에 "
            "`APP_PASSWORD`를 설정한 뒤 앱을 다시 실행하세요."
        )
        st.stop()

    tokens = _tokens()
    request = st.session_state.get("_login_storage_request")
    if request is None and not st.session_state.get("_login_storage_probed"):
        _request("read")
        request = st.session_state["_login_storage_request"]
        st.session_state["_login_storage_probed"] = True

    if request is not None:
        result = _mount_storage(request)
        if result is None:
            token = st.session_state.get("_login_token")
            if tokens.valid(token, password):
                st.session_state["authenticated"] = True
                _render_logout(tokens, token)
                return
            _show_login_form(password, tokens)
            if request.get("action") == "read":
                st.caption("저장된 로그인 상태를 확인하는 중…")
            st.stop()

        st.session_state.pop("_login_storage_request", None)
        if result.get("error"):
            st.warning(
                "로그인 상태를 브라우저에 저장할 수 없습니다. "
                "사이트 저장소를 허용하면 새로고침 후에도 로그인이 유지됩니다."
            )
        else:
            restored = result.get("token")
            if tokens.valid(restored, password):
                st.session_state["_login_token"] = restored

    token = st.session_state.get("_login_token")
    if tokens.valid(token, password):
        st.session_state["authenticated"] = True
        _render_logout(tokens, token)
        return

    st.session_state["authenticated"] = False
    _show_login_form(password, tokens)
    st.stop()
