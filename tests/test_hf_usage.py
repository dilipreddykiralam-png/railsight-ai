import importlib
import json
import sys
import types
import pytest


@pytest.fixture
def counter(monkeypatch, tmp_path):
    state = {
        "schemaVersion": 1,
        "label": "completed analyses",
        "message": "0",
        "color": "blue",
        "cacheSeconds": 300,
        "event_keys": [],
    }
    revision = {"value": 0, "fail_status": None, "conflict_once": False, "read_failure": False}

    class EntryNotFoundError(Exception):
        pass

    class HfHubHTTPError(Exception):
        def __init__(self, status):
            self.response = types.SimpleNamespace(status_code=status)

    class CommitOperationAdd:
        def __init__(self, *, path_in_repo, path_or_fileobj):
            self.path_in_repo = path_in_repo
            self.path_or_fileobj = path_or_fileobj

    class HfApi:
        def __init__(self, token=None):
            self.token = token

        def dataset_info(self, repo_id):
            assert repo_id == "dilipbobby/railsight-ai-usage"
            if revision['read_failure']:
                raise OSError('Offline')
            return types.SimpleNamespace(sha=f"revision-{revision['value']}")

        def create_commit(self, *, operations, parent_commit, **kwargs):
            assert self.token == 'test-token'
            if revision['fail_status']:
                raise HfHubHTTPError(revision['fail_status'])
            if revision['conflict_once']:
                revision['conflict_once'] = False
                state['message'] = '1'
                state['counts'] = {'images': 0, 'videos': 1, 'legacy': 0}
                revision['value'] += 1
                raise HfHubHTTPError(409)
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
    monkeypatch.delenv('RAILSIGHT_USAGE_TOKEN', raising=False)
    yield usage, state, revision
    sys.modules.pop("railreview.hf_usage", None)


def test_missing_token_is_explicit_and_does_not_write(counter):
    usage, state, revision = counter
    assert usage.read_totals() == dict(total=0, images=0, videos=0, legacy=0)
    assert 'Counter not connected' in usage.display_text()
    assert usage.record_completed_analysis('report-1', 'image') is None
    assert revision['value'] == 0


def test_images_videos_deduplication_and_restart(counter, monkeypatch):
    usage, state, _ = counter
    monkeypatch.setenv('RAILSIGHT_USAGE_TOKEN', 'test-token')
    assert usage.record_completed_analysis('report-1', 'image')['total'] == 1
    assert usage.record_completed_analysis('report-1', 'image')['total'] == 1
    assert usage.record_completed_analysis('report-2', 'video') == dict(total=2, images=1, videos=1, legacy=0)
    assert "report-1" not in state["event_keys"]
    assert state['message'] == '2'  # Shields badge total agrees with media totals.
    assert len(state['event_keys']) == 2
    usage = importlib.reload(usage)
    assert usage.read_totals() == dict(total=2, images=1, videos=1, legacy=0)
    assert 'Images analyzed:** 1' in usage.display_text()
    assert 'Videos analyzed:** 1' in usage.display_text()


def test_legacy_total_is_preserved_without_inventing_media_types(counter, monkeypatch):
    usage, state, _ = counter
    state['message'] = '8'
    monkeypatch.setenv('RAILSIGHT_USAGE_TOKEN', 'test-token')
    assert usage.record_completed_analysis('new-video', 'video') == dict(total=9, images=0, videos=1, legacy=8)
    assert '8 earlier runs' in usage.display_text()


def test_concurrent_write_rereads_before_increment(counter, monkeypatch):
    usage, _, revision = counter
    revision['conflict_once'] = True
    monkeypatch.setenv('RAILSIGHT_USAGE_TOKEN', 'test-token')
    assert usage.record_completed_analysis('image-report', 'image') == dict(total=2, images=1, videos=1, legacy=0)


@pytest.mark.parametrize('status', [403, 500, 409])
def test_failed_write_does_not_claim_success_or_change_count(counter, monkeypatch, status):
    usage, state, revision = counter
    monkeypatch.setenv('RAILSIGHT_USAGE_TOKEN', 'test-token')
    usage.read_totals()
    revision['fail_status'] = status
    assert usage.record_completed_analysis('not-saved', 'image') is None
    assert state['message'] == '0' and revision['value'] == 0
    assert 'Counter update failed' in usage.display_text()
    revision['fail_status'] = None
    assert usage.record_completed_analysis('not-saved', 'image')['total'] == 1
    assert 'Counter update failed' not in usage.display_text()


def test_offline_read_retains_saved_total_and_marks_it_stale(counter, monkeypatch):
    usage, _, revision = counter
    monkeypatch.setenv('RAILSIGHT_USAGE_TOKEN', 'test-token')
    usage.record_completed_analysis('report-1', 'image')
    revision['read_failure'] = True
    assert usage.read_totals(cache_seconds=0)['total'] == 1
    assert 'last saved values' in usage.display_text()


def test_corrupt_totals_are_not_overwritten(counter, monkeypatch):
    usage, state, revision = counter
    monkeypatch.setenv('RAILSIGHT_USAGE_TOKEN', 'test-token')
    state['counts'] = dict(images=10, videos=0, legacy=0)
    assert usage.record_completed_analysis('report-1', 'image') is None
    assert revision['value'] == 0 and state['counts']['images'] == 10
