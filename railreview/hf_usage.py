"""Persistent, privacy-preserving count of completed hosted analyses.

Only a completed-analysis total and short-lived hashes of random report IDs are
stored in the configured public Hugging Face Dataset repository. Uploaded media,
model output, reviewer details, IP addresses, and user identities are not sent.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import logging
import os
import threading
import time

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

LOGGER = logging.getLogger(__name__)
COUNTER_REPO = os.getenv("RAILSIGHT_USAGE_REPO", "dilipbobby/railsight-ai-usage")
COUNTER_FILE = "usage.json"
MAX_RETRIES = 5
MAX_RECENT_EVENT_KEYS = 256
_lock = threading.Lock()
_cached_count: int | None = None
_cached_at = 0.0


def _token() -> str | None:
    return os.getenv("RAILSIGHT_USAGE_TOKEN")


def _api(token: str | None) -> HfApi:
    return HfApi(token=token)


def _read_at_revision(api: HfApi, token: str, revision: str) -> dict:
    try:
        path = hf_hub_download(
            repo_id=COUNTER_REPO,
            repo_type="dataset",
            filename=COUNTER_FILE,
            revision=revision,
            token=token,
        )
    except EntryNotFoundError:
        return {"schemaVersion": 1, "label": "completed analyses", "message": "0",
                "color": "blue", "cacheSeconds": 300, "event_keys": []}
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    count = value.get("message", "0")
    if not str(count).isdigit():
        raise ValueError("Usage counter has an invalid count")
    keys = value.get("event_keys", [])
    if not isinstance(keys, list) or any(not isinstance(item, str) for item in keys):
        raise ValueError("Usage counter has invalid event keys")
    return {"schemaVersion": 1, "label": "completed analyses", "message": str(int(count)),
            "color": "blue", "cacheSeconds": 300, "event_keys": keys[-MAX_RECENT_EVENT_KEYS:]}


def read_count(*, cache_seconds: int = 300) -> int | None:
    """Read the public total; writing still requires the Space's fine-grained token."""
    global _cached_count, _cached_at
    token = _token()
    with _lock:
        if _cached_count is not None and time.monotonic() - _cached_at < cache_seconds:
            return _cached_count
        try:
            api = _api(token)
            revision = api.dataset_info(COUNTER_REPO).sha
            value = _read_at_revision(api, token, revision)
            _cached_count = int(value["message"])
            _cached_at = time.monotonic()
            return _cached_count
        except Exception as exc:  # Counter outages must not block image analysis.
            LOGGER.warning("Persistent usage counter read failed (%s)", type(exc).__name__)
            return _cached_count


def record_completed_analysis(report_id: str) -> int | None:
    """Atomically increment the public total once for this random report ID."""
    global _cached_count, _cached_at
    token = _token()
    if not token or not report_id:
        return None
    event_key = sha256(report_id.encode("utf-8")).hexdigest()
    with _lock:
        api = _api(token)
        for _ in range(MAX_RETRIES):
            try:
                info = api.dataset_info(COUNTER_REPO)
                value = _read_at_revision(api, token, info.sha)
                keys = value["event_keys"]
                if event_key in keys:
                    _cached_count = int(value["message"])
                    _cached_at = time.monotonic()
                    return _cached_count
                count = int(value["message"]) + 1
                updated = {
                    "schemaVersion": 1,
                    "label": "completed analyses",
                    "message": str(count),
                    "color": "blue",
                    "cacheSeconds": 300,
                    "event_keys": (keys + [event_key])[-MAX_RECENT_EVENT_KEYS:],
                }
                api.create_commit(
                    repo_id=COUNTER_REPO,
                    repo_type="dataset",
                    operations=[CommitOperationAdd(
                        path_in_repo=COUNTER_FILE,
                        path_or_fileobj=BytesIO(json.dumps(updated, separators=(",", ":")).encode()),
                    )],
                    commit_message="Update anonymous completed-analysis total",
                    parent_commit=info.sha,
                )
                _cached_count = count
                _cached_at = time.monotonic()
                return count
            except HfHubHTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (409, 412):
                    continue  # Another visitor updated the total; reread and retry.
                LOGGER.warning("Persistent usage counter write failed (HTTP %s)", status)
                return _cached_count
            except Exception as exc:  # Analysis remains available if the counter is down.
                LOGGER.warning("Persistent usage counter write failed (%s)", type(exc).__name__)
                return _cached_count
        LOGGER.warning("Persistent usage counter write conflicted repeatedly")
        return _cached_count


def display_text(count: int | None = None) -> str:
    """Text to show in the app without implying a unique-person count."""
    if count is None:
        count = read_count()
    if count is None:
        return "**Completed analyses:** counting is being connected."
    return f"**Completed analyses:** {count:,} · One complete image or video run counts once; this is not a unique-user count."
