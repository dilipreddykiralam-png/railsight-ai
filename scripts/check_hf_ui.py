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
app = runpy.run_path(str(ROOT / 'dist/huggingface/app.py'))
namespace = app['analyze'].__globals__
fixture = json.loads((ROOT / 'examples/outputs/donetsk-train.json').read_text())['model_assessment']
namespace['infer'] = lambda *args: json.dumps(fixture)
result = app['analyze'](str(ROOT / 'examples/images/donetsk-train.jpg'), 'Image', 2, False)
record = result[0]
assert record['prediction'] and not record['error'], record
assert record['generation_settings']['structured_format'] is False
assert record['generation_settings']['engine'] == 'transformers'
assert result[1] is True
assert len(result[3]) == 1
reviewed = app['save_review'](record, 'fixture-reviewer', result[6], 'Fixture test only', *result[7:12])[0]
assert reviewed['human_verification']['status'] == 'confirmed'
assert record['human_verification']['status'] == 'pending'
assert reviewed['asset_verification']['findings']
# Exercise video sampling and grouped presentation with fixture inference.
import cv2, tempfile
video_path = str(Path(tempfile.mkdtemp()) / 'fixture.mp4')
image = cv2.imread(str(ROOT / 'examples/images/donetsk-train.jpg'))
image = cv2.resize(image, (320, 240))
writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (320, 240))
for _ in range(10): writer.write(image)
writer.release()
video = app['analyze'](video_path, 'Video', 2, True)
assert video[0]['prediction'] and len(video[0]['frames']) == 2
assert video[0]['video_presentation']['selected_sample_numbers']
assert len(video[3]) == 2
partial = copy.deepcopy(video[0])
partial['frames'][1].update(prediction=None, error='Quota exhausted')
assert 'PARTIAL ANALYSIS: 1 of 2' in app['summary'](partial)
# Exercise a failure: no fabricated prediction or successful-test increment.
namespace['infer'] = lambda *args: 'not valid JSON'
failed = app['analyze'](str(ROOT / 'examples/images/donetsk-train.jpg'), 'Image', 2, False)
assert failed[0]['prediction'] is None
assert failed[1] is False
assert '2 completed tests' in failed[-2]
assert Path(result[5]).exists()
assert app['demo'].get_config_file()['components']
print('Hosted UI builds; image analysis, metadata, independent review, export and failure counting checks passed with fixture inference.')
