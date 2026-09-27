import importlib
import json
import sys
import types


def test_public_count_is_readable_without_write_token_and_events_are_deduplicated(monkeypatch, tmp_path):
    state = {
        "schemaVersion": 1,
        "label": "completed analyses",
        "message": "0",
        "color": "blue",
        "cacheSeconds": 300,
        "event_keys": [],
    }
    revision = {"value": 0}

    class EntryNotFoundError(Exception):
        pass

    class HfHubHTTPError(Exception):
        pass

    class CommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class HfApi:
        def __init__(self, token=None):
            self.token = token

        def dataset_info(self, repo_id):
            assert repo_id == "dilipbobby/railsight-ai-usage"
            return types.SimpleNamespace(sha=f"revision-{revision['value']}")

        def create_commit(self, *, operations, parent_commit, **kwargs):
            assert parent_commit == f"revision-{revision['value']}"
            payload = operations[0].path_or_fileobj.read()
            state.update(json.loads(payload))
            revision["value"] += 1

    def download(*, repo_id, repo_type, filename, revision, token):
        assert repo_id == "dilipbobby/railsight-ai-usage"
        assert repo_type == "dataset" and filename == "usage.json"
        path = tmp_path / "usage.json"
        path.write_text(json.dumps(state), encoding="utf-8")
        return str(path)

    hub = types.ModuleType("huggingface_hub")
    hub.CommitOperationAdd = CommitOperationAdd
    hub.HfApi = HfApi
    hub.hf_hub_download = download
    errors = types.ModuleType("huggingface_hub.errors")
    errors.EntryNotFoundError = EntryNotFoundError
    errors.HfHubHTTPError = HfHubHTTPError
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    sys.modules.pop("railreview.hf_usage", None)
    usage = importlib.import_module("railreview.hf_usage")
    monkeypatch.setattr(usage, "COUNTER_REPO", "dilipbobby/railsight-ai-usage")

    monkeypatch.setattr(usage, "_token", lambda: None)
    assert usage.read_count() == 0
    assert usage.display_text().startswith("**Completed analyses:** 0")
    assert usage.record_completed_analysis("report-1") is None

    monkeypatch.setattr(usage, "_token", lambda: "fine-grained-write-token")
    assert usage.record_completed_analysis("report-1") == 1
    assert usage.record_completed_analysis("report-1") == 1
    assert usage.read_count() == 1
    assert state["message"] == "1"
    assert "report-1" not in state["event_keys"]

    sys.modules.pop("railreview.hf_usage", None)
