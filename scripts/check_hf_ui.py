"""Check hosted UI and callbacks with fixture inference; does not test GPU quality."""
import copy
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
class Loader:
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def eval(self): return self
sys.modules['spaces'] = SimpleNamespace(GPU=lambda **kwargs: lambda fn: fn)
sys.modules['torch'] = SimpleNamespace(bfloat16='bfloat16')
sys.modules['transformers'] = SimpleNamespace(Qwen2_5_VLForConditionalGeneration=Loader, AutoProcessor=Loader)
sys.modules['qwen_vl_utils'] = SimpleNamespace(process_vision_info=lambda messages: ([], []))
# Mock the counter before app construction: no public counter reads or writes.
counts = dict(total=0, images=0, videos=0, legacy=0)
counted_reports = []
def record_completed(report_id, media_type):
    counted_reports.append((report_id, media_type))
    counts['total'] += 1
    counts['images' if media_type == 'image' else 'videos'] += 1
    return dict(counts)
sys.modules['railreview.hf_usage'] = SimpleNamespace(
    record_completed_analysis=record_completed,
    display_text=lambda totals=None: '**Completed analyses:** ' + str((totals or counts)['total']))
app = runpy.run_path(str(ROOT / 'dist/huggingface/app.py'))
namespace = app['analyze'].__globals__
fixture = json.loads((ROOT / 'examples/outputs/donetsk-train.json').read_text())['model_assessment']
namespace['infer'] = lambda *args: json.dumps(fixture)
result = app['analyze'](str(ROOT / 'examples/images/donetsk-train.jpg'), 'Image', 2)
record = result[0]
assert record['prediction'] and not record['error'], record
assert record['generation_settings']['structured_format'] is False
assert record['generation_settings']['engine'] == 'transformers'
assert len(result[2]) == 1
assert counted_reports == [(record['id'], 'image')]
assert record['usage_count_saved'] is True
reviewed = app['save_review'](record, 'fixture-reviewer', result[5], 'Fixture test only', *result[6:11])[0]
assert reviewed['human_verification']['status'] == 'confirmed'
assert record['human_verification']['status'] == 'pending'
assert reviewed['asset_verification']['findings']
assert counts['total'] == 1  # Human verification is not a new analysis.
# Exercise video sampling and grouped presentation with fixture inference.
import cv2, tempfile
video_path = str(Path(tempfile.mkdtemp()) / 'fixture.mp4')
image = cv2.imread(str(ROOT / 'examples/images/donetsk-train.jpg'))
image = cv2.resize(image, (320, 240))
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (320, 240))
for _ in range(10): writer.write(image)
writer.release()
video = app['analyze'](video_path, 'Video', 2)
assert video[0]['prediction'] and len(video[0]['frames']) == 2
assert video[0]['video_presentation']['selected_sample_numbers']
assert len(video[2]) == 2
assert counts == dict(total=2, images=1, videos=1, legacy=0)
assert counted_reports[-1] == (video[0]['id'], 'video')
partial = copy.deepcopy(video[0])
partial['frames'][1].update(prediction=None, error='Quota exhausted')
assert 'PARTIAL ANALYSIS: 1 of 2' in app['summary'](partial)
namespace['run_video'] = lambda *a, **k: copy.deepcopy(partial)
partial_result = app['analyze'](video_path, 'Video', 2)
assert partial_result[0]['usage_count_saved'] is False
assert counts['total'] == 2
# Exercise a failure: no fabricated prediction or successful-test increment.
namespace['infer'] = lambda *args: 'not valid JSON'
failed = app['analyze'](str(ROOT / 'examples/images/donetsk-train.jpg'), 'Image', 2)
assert failed[0]['prediction'] is None
assert counts['total'] == 2
assert failed[0]['usage_count_saved'] is False
assert 'completed analyses' in failed[-2].lower()
assert Path(result[4]).exists()
assert app['demo'].get_config_file()['components']
# A counter outage must not discard a valid analysis or claim it was counted.
namespace['infer'] = lambda *args: json.dumps(fixture)
namespace['record_completed_analysis'] = lambda *args: None
outage = app['analyze'](str(ROOT / 'examples/images/donetsk-train.jpg'), 'Image', 2)
assert outage[0]['prediction'] and not outage[0]['error']
assert outage[0]['usage_count_saved'] is False
print('Hosted UI builds; image analysis, metadata, independent review, export and failure counting checks passed with fixture inference.')
