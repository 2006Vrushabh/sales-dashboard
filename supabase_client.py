import os
import streamlit as st
from supabase import create_client, Client


@st.cache_resource
def get_client() -> Client:
    # Render / other hosting platforms
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    # Local development
    if not url or not key:
        try:
            url = st.secrets["SUPABASE_URL"]
            key = st.secrets["SUPABASE_KEY"]
        except Exception:
            st.error(
                "Supabase credentials are missing. "
                "Configure SUPABASE_URL and SUPABASE_KEY."
            )
            st.stop()

    return create_client(url, key)