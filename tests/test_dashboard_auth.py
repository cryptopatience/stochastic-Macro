import os
from pathlib import Path
import time

import pytest

from dashboard_auth import (
    LoginTokens,
    TOKEN_TTL_SECONDS,
    passwords_match,
    read_app_password,
)

ROOT = Path(__file__).resolve().parents[1]


def test_component_uses_flat_index_html():
    html = ROOT / "auth_component" / "index.html"
    assert html.is_file()
    text = html.read_text(encoding="utf-8")
    assert "sessionStorage" in text
    assert "scgt-dashboard-session-v1" in text
    assert not (ROOT / "auth_component" / "auth_component" / "index.html").exists()


def test_passwords_match_is_length_safe():
    assert passwords_match("secret", "secret")
    assert not passwords_match("nope", "secret")
    assert not passwords_match("", "secret")
    assert not passwords_match("secret", "")


def test_login_tokens_issue_validate_and_revoke():
    tokens = LoginTokens()
    token = tokens.issue("correct-horse")
    assert tokens.valid(token, "correct-horse")
    assert not tokens.valid(token, "wrong-password")
    assert not tokens.valid("missing", "correct-horse")
    assert not tokens.valid(None, "correct-horse")
    tokens.revoke(token)
    assert not tokens.valid(token, "correct-horse")


def test_login_tokens_expire(monkeypatch):
    tokens = LoginTokens()
    now = 1_000.0
    monkeypatch.setattr(time, "time", lambda: now)
    token = tokens.issue("pw")
    assert tokens.valid(token, "pw")
    monkeypatch.setattr(time, "time", lambda: now + TOKEN_TTL_SECONDS + 1)
    assert not tokens.valid(token, "pw")


def test_read_app_password_prefers_secrets(monkeypatch):
    class FakeSecrets(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    import dashboard_auth as auth

    monkeypatch.setattr(auth.st, "secrets", FakeSecrets(APP_PASSWORD="from-secrets"))
    monkeypatch.setenv("APP_PASSWORD", "from-env")
    assert read_app_password() == "from-secrets"


def test_read_app_password_uses_env_and_fails_closed(monkeypatch):
    class ExplodingSecrets:
        def get(self, key, default=None):
            raise RuntimeError("no secrets file")

    import dashboard_auth as auth

    monkeypatch.setattr(auth.st, "secrets", ExplodingSecrets())
    monkeypatch.setenv("APP_PASSWORD", "from-env")
    assert read_app_password() == "from-env"
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    assert read_app_password() is None


def test_read_app_password_rejects_blank(monkeypatch):
    class FakeSecrets(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    import dashboard_auth as auth

    monkeypatch.setattr(auth.st, "secrets", FakeSecrets(APP_PASSWORD="   "))
    monkeypatch.setenv("APP_PASSWORD", "")
    assert read_app_password() is None


def test_pages_call_require_login_instead_of_session_flag():
    files = [
        ROOT / "app.py",
        ROOT / "pages" / "1_SSO.py",
        ROOT / "pages" / "2_Macro.py",
        ROOT / "pages" / "3_AI_종합분석.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "from dashboard_auth import require_login" in text
        assert "require_login()" in text
        assert 'st.session_state.get("authenticated")' not in text
        assert 'secrets.get("APP_PASSWORD", "1234")' not in text


@pytest.fixture
def isolated_env(monkeypatch):
    monkeypatch.delenv("APP_PASSWORD", raising=False)
    return monkeypatch


def test_app_fails_closed_without_password(isolated_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=10)
    at.run()
    assert at.error
    assert any("APP_PASSWORD" in str(item.value) for item in at.error)
    assert not at.session_state.get("authenticated")


def _click_labeled_button(at, label):
    matches = [btn for btn in at.button if label in str(btn.label)]
    assert matches, f"button {label!r} not found in {[btn.label for btn in at.button]}"
    return matches[0].click().run()


def test_app_login_wrong_then_right_password_then_logout(isolated_env):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=10)
    at.secrets["APP_PASSWORD"] = "s3cret"
    at.run()
    assert at.text_input, "login form should render even while storage restore is pending"
    assert "통합 투자 대시보드" in "\n".join(str(item.value) for item in at.markdown)

    at.text_input[0].set_value("nope")
    _click_labeled_button(at, "로그인")
    assert any("틀렸습니다" in str(item.value) for item in at.error)
    assert not at.session_state.get("authenticated")

    at.text_input[0].set_value("s3cret")
    _click_labeled_button(at, "로그인")
    assert at.session_state.get("authenticated") is True
    assert at.session_state.get("_login_token")
    page_html = "\n".join(str(item.value) for item in at.markdown)
    assert "투자 리서치 대시보드" in page_html

    logout = [btn for btn in at.sidebar.button if "로그아웃" in str(btn.label)]
    assert logout
    logout[0].click().run()
    assert not at.session_state.get("authenticated")
    assert at.text_input, "logout should return to the login form"
