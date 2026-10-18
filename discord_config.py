"""Discord webhook URL lookup.

Reads DISCORD_WEBHOOK_URL from the environment first, then from Streamlit
secrets under the same name. A missing value is not an error: callers should
skip the send and keep running.
"""

from __future__ import annotations

import os

DISCORD_WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"
DISCORD_WEBHOOK_SECRET = "DISCORD_WEBHOOK_URL"


def discord_webhook_url() -> str:
    """Return the configured webhook URL, or '' when it is unset."""
    env_url = os.environ.get(DISCORD_WEBHOOK_ENV, "").strip()
    if env_url:
        return env_url
    try:
        import streamlit as st

        value = st.secrets.get(DISCORD_WEBHOOK_SECRET, "")
    except Exception:
        return ""
    if value is None:
        return ""
    return str(value).strip()
