"""
database.py
------------
Small, reusable functions for reading/writing the "datasets" table.

Every function here filters by user_id even though Supabase Row Level
Security already enforces this on the server -- doing it on both sides is
just good practice and makes the queries easier to read.

Kept separate from app.py purely so app.py doesn't get too long, per the
project brief's own instruction to split files only once the main file
gets unwieldy.
"""

from typing import Optional, List
from supabase_client import get_client


def get_dataset_count(user_id: str) -> int:
    """How many datasets this user has saved. Used on the dashboard."""
    client = get_client()
    result = client.table("datasets").select("id", count="exact").eq("user_id", user_id).execute()
    return result.count or 0


def get_latest_dataset(user_id: str) -> Optional[dict]:
    """The user's most recently saved dataset, or None if they have none yet."""
    client = get_client()
    result = (
        client.table("datasets")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return result.data[0] if result.data else None


def get_all_datasets(user_id: str) -> List[dict]:
    """Every dataset this user has saved, newest first. Used on the History page."""
    client = get_client()
    result = (
        client.table("datasets")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data


def save_dataset_record(user_id: str, record: dict) -> None:
    """Insert one row into the datasets table. `record` holds the metadata columns."""
    client = get_client()
    client.table("datasets").insert({**record, "user_id": user_id}).execute()
