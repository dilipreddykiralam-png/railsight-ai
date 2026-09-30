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
_cached_totals: dict | None = None
_cached_at = 0.0
_read_failed = False
_write_problem = False


def _token() -> str | None:
    return os.getenv("RAILSIGHT_USAGE_TOKEN")


def _api(token: str | bool) -> HfApi:
    return HfApi(token=token)


def _read_at_revision(token: str | bool, revision: str) -> dict:
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
                "color": "blue", "cacheSeconds": 300,
                "counts": {"images": 0, "videos": 0, "legacy": 0}, "event_keys": []}
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    count = value.get("message", "0")
    if not str(count).isdigit():
        raise ValueError("Usage counter has an invalid count")
    keys = value.get("event_keys", [])
    if not isinstance(keys, list) or any(not isinstance(item, str) for item in keys):
        raise ValueError("Usage counter has invalid event keys")
    # Older totals did not distinguish images and videos. Preserve them without
    # inventing a split when upgrading the public counter file.
    counts = value.get("counts", {"images": 0, "videos": 0, "legacy": int(count)})
    if (not isinstance(counts, dict) or set(counts) != {"images", "videos", "legacy"}
            or any(type(number) is not int or number < 0 for number in counts.values())
            or sum(counts.values()) != int(count)):
        raise ValueError("Usage counter has inconsistent media totals")
    return {"schemaVersion": 1, "label": "completed analyses", "message": str(int(count)),
            "color": "blue", "cacheSeconds": 300, "counts": counts,
            "event_keys": keys[-MAX_RECENT_EVENT_KEYS:]}


def _totals(value: dict) -> dict:
    return {"total": int(value["message"]), **value["counts"]}


def read_totals(*, cache_seconds: int = 60) -> dict | None:
    """Read saved public totals, even if the Space has no working write token."""
    global _cached_totals, _cached_at, _read_failed
    with _lock:
        if _cached_totals is not None and time.monotonic() - _cached_at < cache_seconds:
            return dict(_cached_totals)
        try:
            api = _api(False)
            revision = api.dataset_info(COUNTER_REPO).sha
            value = _read_at_revision(False, revision)
            _cached_totals = _totals(value)
            _cached_at = time.monotonic()
            _read_failed = False
        except Exception as exc:  # Counter outages must not block image analysis.
            LOGGER.warning("Persistent usage counter read failed (%s)", type(exc).__name__)
            _read_failed = True
        return dict(_cached_totals) if _cached_totals is not None else None


def record_completed_analysis(report_id: str, media_type: str) -> dict | None:
    """Atomically increment the public total once for this random report ID."""
    global _cached_totals, _cached_at, _read_failed, _write_problem
    if media_type not in ("image", "video"):
        raise ValueError("Expected image or video for usage counting")
    token = _token()
    if not token or not report_id:
        return None
    event_key = sha256(report_id.encode("utf-8")).hexdigest()
    with _lock:
        for _ in range(MAX_RETRIES):
            try:
                api = _api(token)
                info = api.dataset_info(COUNTER_REPO)
                value = _read_at_revision(token, info.sha)
                keys = value["event_keys"]
                if event_key in keys:
                    _cached_totals = _totals(value)
                    _cached_at = time.monotonic()
                    _write_problem = _read_failed = False
                    return dict(_cached_totals)
                count = int(value["message"]) + 1
                counts = dict(value["counts"])
                counts["images" if media_type == "image" else "videos"] += 1
                updated = {
                    "schemaVersion": 1,
                    "label": "completed analyses",
                    "message": str(count),
                    "color": "blue",
                    "cacheSeconds": 300,
                    "counts": counts,
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
                _cached_totals = _totals(updated)
                _cached_at = time.monotonic()
                _write_problem = _read_failed = False
                return dict(_cached_totals)
            except HfHubHTTPError as exc:
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (409, 412):
                    continue  # Another visitor updated the total; reread and retry.
                LOGGER.warning("Persistent usage counter write failed (HTTP %s)", status)
                _write_problem = True
                return None
            except Exception as exc:  # Analysis remains available if the counter is down.
                LOGGER.warning("Persistent usage counter write failed (%s)", type(exc).__name__)
                _write_problem = True
                return None
        LOGGER.warning("Persistent usage counter write conflicted repeatedly")
        _write_problem = True
        return None


def display_text(totals: dict | None = None) -> str:
    """Text to show in the app without implying a unique-person count."""
    if totals is None:
        totals = read_totals()
    if totals is None:
        text = "**Completed analyses:** totals unavailable."
    else:
        text = (f"**Images analyzed:** {totals['images']:,} · "
                f"**Videos analyzed:** {totals['videos']:,} · "
                f"**Total completed analyses:** {totals['total']:,}")
        if totals['legacy']:
            text += f" · Includes {totals['legacy']:,} earlier runs without a media-type breakdown."
    if not _token():
        text += "\n\n**Counter not connected:** the owner must finish setup before new analyses can be saved."
    elif _write_problem:
        text += "\n\n**Counter update failed:** the latest completed analysis was not added. Showing previously saved totals."
    elif _read_failed:
        text += "\n\n**Counter temporarily unavailable:** any displayed totals are the last saved values."
    text += "\n\nOne successfully completed image or video counts once. Video frames, failed or partial runs, and page visits are not counted. Totals refresh automatically."
    return text
