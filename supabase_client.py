"""
supabase_client.py
-------------------
Creates one shared connection to Supabase.

@st.cache_resource means Streamlit builds this client ONCE and reuses it on
every rerun, instead of reconnecting every time the user clicks something
(Streamlit re-runs the whole script top-to-bottom on every interaction).

Credentials come from Streamlit secrets, never from code, so nothing secret
is ever committed to GitHub. Locally this reads .streamlit/secrets.toml; on
Streamlit Community Cloud it reads the secrets you paste into the app's
settings.
"""

import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_client() -> Client:
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        st.error(
            "Supabase credentials are missing. Add SUPABASE_URL and "
            "SUPABASE_KEY to .streamlit/secrets.toml (see README)."
        )
        st.stop()
    return create_client(url, key)
