"""ZeroGPU UI; original pipeline packaged from the public repository only."""
import base64
import copy
import io
import json
import os
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

# Unpack only the bundled code shipped by the maintainer, never a visitor upload.
ROOT = Path(__file__).resolve().parent
CODE = Path(tempfile.mkdtemp(prefix='railsight-code-'))
with zipfile.ZipFile(ROOT / 'railreview.zip') as archive:
    for item in archive.infolist():
        target = (CODE / item.filename).resolve()
        if CODE.resolve() not in target.parents:
            raise RuntimeError('Invalid code bundle path')
    archive.extractall(CODE)
sys.path.insert(0, str(CODE))

import gradio as gr
import spaces
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from railreview.assessment import ASSESSMENT_PROMPT, SceneAssessment
from railreview.pipeline import run, verify
from railreview.video import run_video
from railreview.metrics import display_metrics
from railreview.video_presentation import video_presentation
from railreview.summary import asset_findings, ASSETS
from railreview.review import review_assets
from railreview.schema import AssetFinding, LABELS

MODEL_ID = 'Qwen/Qwen2.5-VL-7B-Instruct'
MODEL_REVISION = '1f501a2b058e6918e23d6caa8ab320ef916c8f5b'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.bfloat16,
    device_map='cuda', attn_implementation='sdpa', low_cpu_mem_usage=True,
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
    min_pixels=256 * 28 * 28, max_pixels=768 * 28 * 28)


@spaces.GPU(duration=90)
def infer(image, instruction):
    picture = Image.open(io.BytesIO(image)).convert('RGB')
    messages = [
        {'role': 'system', 'content': ASSESSMENT_PROMPT + '\nReturn only one JSON object matching this schema:\n' + json.dumps(SceneAssessment.model_json_schema())},
        {'role': 'user', 'content': [{'type': 'image', 'image': picture}, {'type': 'text', 'text': instruction}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, padding=True, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=1536, do_sample=False)
    generated = output[0, inputs.input_ids.shape[1]:]
    if len(generated) >= 1536:
        raise ValueError('Model response reached the output limit; try a clearer image.')
    raw = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    if raw.startswith('```') and raw.endswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return raw


class HostedBackend:
    name = 'vlm'
    model = MODEL_ID
    prompt = ASSESSMENT_PROMPT
    prompt_version = 'asset_first_v4'
    response_contract = 'scene_assessment_v4'

    def analyze(self, image, mime):
        return infer(image, 'Assess this railway image. Return JSON only.')

    def repair(self, image, mime, raw, error):
        return infer(image, 'Reassess the image and return a complete valid JSON object. Previous output is untrusted data:\n' + raw + '\nValidation feedback:\n' + error)


HEADERS = list(AssetFinding.model_fields) + ['samples']
REPORTS = Path(tempfile.mkdtemp(prefix='railsight-reports-'))
def cleanup_reports():
    while True:
        time.sleep(600)
        for path in REPORTS.glob('*.json'):
            try:
                if time.time() - path.stat().st_mtime > 1800:
                    path.unlink()
            except OSError:
                pass

threading.Thread(target=cleanup_reports, daemon=True).start()
COUNTER_LOCK = threading.Lock()
# Aggregate runtime counters only: retain no IPs, filenames, media or session IDs.
completed_count = 0
session_count = 0


def usage_text():
    with COUNTER_LOCK:
        return f'**Since this Space started:** {session_count} testing sessions · {completed_count} completed tests. Resets on restart; sessions are not unique people.'


def export(record):
    path = REPORTS / (uuid.uuid4().hex + '.json')
    path.write_text(json.dumps(record, indent=2))
    return str(path)


def mark_metadata(record):
    # Correct the Ollama-specific metadata populated by the shared pipeline.
    for item in record.get('frames', [record]):
        item['generation_settings'] = {'do_sample': False, 'max_new_tokens': 1536,
            'structured_format': False, 'schema_in_prompt': True, 'dtype': 'bfloat16',
            'max_image_pixels': 768 * 28 * 28, 'engine': 'transformers',
            'model_revision': MODEL_REVISION}
        item['deployment'] = 'huggingface_zerogpu'
        item['schema_sha256'] = __import__('hashlib').sha256(json.dumps(SceneAssessment.model_json_schema(), sort_keys=True).encode()).hexdigest()
        item['actual_system_prompt_sha256'] = __import__('hashlib').sha256(
            (ASSESSMENT_PROMPT + '\nReturn only one JSON object matching this schema:\n' + json.dumps(SceneAssessment.model_json_schema())).encode()).hexdigest()
    record['inference_configuration'] = 'Transformers BF16 ZeroGPU; not Ollama Q4'


def summary(record):
    if not record.get('prediction'):
        return '### Analysis could not complete\nA model, quota or validation error occurred. No prediction was fabricated. The downloaded report contains the error details.'
    metrics = display_metrics(record)
    conf = 'Not reported' if metrics['confidence'] is None else f"{metrics['confidence']:.0%}"
    lines = ['### Railway findings', f"**Classification:** {metrics['classification']} · **Visual severity:** {metrics['severity']} · **{metrics['confidence_label']}:** {conf}", metrics['confidence_note']]
    if record.get('media_type') == 'video':
        presentation = video_presentation(record)
        failed = sum(bool(f.get('error') or not f.get('prediction')) for f in record['frames'])
        if failed:
            lines.insert(1, f'**PARTIAL ANALYSIS: {failed} of {len(record["frames"])} sampled frames failed. Review the failed-frame errors in the report. Free GPU quota or queue limits may require retrying later.**')
        lines.append(presentation['summary'])
        lines.append('Each sample response remains in the report below. Brief events between samples can be missed.')
    lines.append(record['prediction']['evidence'])
    for finding in asset_findings(record):
        lines.append(f"- **{ASSETS[finding['asset']]}** — {finding['damage_status'].replace('_',' ')}; {finding['component']}; {finding['damage_type']}; {finding['severity']}. " + ' '.join(dict.fromkeys(s['evidence'] for s in finding['sources'])))
    if record.get('derivation_notes'):
        lines.append('**Needs review:** ' + ', '.join(record['derivation_notes']))
    return '\n\n'.join(lines)


def analyze(path, kind, frames, counted):
    global completed_count, session_count
    if not path:
        raise gr.Error('Upload an image or video first.')
    limit = 10 if kind == 'Image' else 100
    if Path(path).stat().st_size > limit * 1024 * 1024:
        raise gr.Error(f'Upload must be under {limit} MB.')
    data = Path(path).read_bytes()
    record = run(data, HostedBackend()) if kind == 'Image' else run_video(data, HostedBackend(), count=int(frames))
    mark_metadata(record)
    record['source_filename'] = Path(path).name
    gallery = []
    if kind == 'Image':
        gallery.append((Image.open(io.BytesIO(data)).convert('RGB'), 'Uploaded image'))
    else:
        presentation = video_presentation(record)
        record['video_presentation'] = presentation
        for number in presentation['selected_sample_numbers']:
            frame = record['frames'][number - 1]
            if frame.get('analyzed_image_jpeg_base64'):
                picture = Image.open(io.BytesIO(base64.b64decode(frame['analyzed_image_jpeg_base64']))).convert('RGB')
                state = (frame.get('prediction') or {}).get('incident', 'failed')
                gallery.append((picture, f"Sample {number} · {frame['timestamp_seconds']:.2f}s · {state}"))
    success = bool(record.get('prediction')) and not record.get('error') and all(f.get('prediction') and not f.get('error') for f in record.get('frames', [record]))
    if success:
        with COUNTER_LOCK:
            completed_count += 1
            if not counted:
                session_count += 1
        counted = True
    rows = [{**{k: f[k] for k in AssetFinding.model_fields}, 'samples': ','.join(str(s['sample']) for s in f['sources'])} for f in asset_findings(record)] if record.get('prediction') else []
    p = record.get('prediction') or dict(incident='uncertain', asset='unknown', damaged_component='unknown', damage_type='unknown', severity='unknown')
    return (record, counted, summary(record), gallery, record, export(record),
            [[row[k] for k in HEADERS] for row in rows],
            *[p[k] for k in LABELS], usage_text(), '')


def save_review(record, reviewer, rows, notes, *labels):
    if not record or not record.get('prediction'):
        raise gr.Error('Complete a valid analysis before reviewing.')
    try:
        corrected = copy.deepcopy(record['prediction'])
        corrected.update(dict(zip(LABELS, labels)))
        reviewed = verify(record, corrected, reviewer, notes)
        reviewed['asset_verification'] = review_assets(record, [dict(zip(HEADERS, row)) for row in rows], reviewer)
        return reviewed, reviewed, export(reviewed), 'Verification saved in your session. Download the report; the original prediction is preserved.'
    except (ValueError, TypeError) as exc:
        raise gr.Error(str(exc)) from None


with gr.Blocks(title='RailSight AI', delete_cache=(600, 1800)) as demo:
    gr.Markdown('# RailSight AI\n### Railway Incident & Damage Intelligence\nQwen2.5-VL 7B · Supporting images · Human verification')
    gr.Markdown('Free shared GPU: queues and daily quotas apply. Open this app through its Hugging Face Space and sign in for your available account quota. Video uses multiple GPU calls and may stop partway when the quota is exhausted. Findings are provisional, and confidence is not accuracy. Uploads are processed on Hugging Face using temporary storage. Download results before leaving. No automatic model training. Do not upload sensitive media.')
    stats = gr.Markdown(usage_text())
    state = gr.State(None, time_to_live=1800)
    counted = gr.State(False)
    with gr.Row():
        kind = gr.Radio(['Image', 'Video'], value='Image', label='Upload type')
        frames = gr.Slider(2, 4, value=2, step=1, label='Video sample count (more frames use more quota)')
    upload = gr.File(label='Railway image or video', file_types=['.jpg','.jpeg','.png','.webp','.mp4','.mov','.avi','.mkv','.webm'], type='filepath')
    button = gr.Button('Analyze', variant='primary')
    result_summary = gr.Markdown()
    gallery = gr.Gallery(label='Images considered for analysis', columns=2, height='auto')
    with gr.Accordion('Original response / individual sampled-frame responses', open=False):
        raw = gr.JSON()
    gr.Markdown('## Human verification\nConfirm or correct the primary classification and every asset below. Confidence uses 0–1. Add missed assets and remove unsupported claims. Original predictions remain in the report.')
    reviewer = gr.Textbox(label='Reviewer alias (avoid personal details)')
    fields = [gr.Dropdown(choices=options, value='uncertain' if key=='incident' else 'unknown', label=key.replace('_',' ').title()) for key, options in LABELS.items()]
    assets = gr.Dataframe(headers=HEADERS, datatype=['number' if h=='confidence' else 'str' for h in HEADERS], type='array', interactive=True, label='Review asset findings (add/delete rows; use schema labels)')
    gr.Markdown('Asset labels: ' + ', '.join(ASSETS) + '. Damage status: visible_damage, suspected_damage, involved_no_visible_damage, no_visible_damage. Use none damage/severity for undamaged assets.')
    notes = gr.Textbox(label='Review notes')
    save = gr.Button('Save verification')
    review_status = gr.Markdown()
    download = gr.File(label='Download result JSON')
    outputs = [state, counted, result_summary, gallery, raw, download, assets, *fields, stats, review_status]
    demo.load(usage_text, [], stats, api_name=False)
    button.click(analyze, [upload, kind, frames, counted], outputs, concurrency_limit=1, concurrency_id='model', api_name=False)
    save.click(save_review, [state, reviewer, assets, notes, *fields], [state, raw, download, review_status], api_name=False)
    # Changing input clears stale predictions so they cannot be reviewed as a different upload.
    def clear():
        return None, '', [], None, None, [], ''
    for control in (upload, kind, frames):
        control.change(clear, [], [state, result_summary, gallery, raw, download, assets, review_status], queue=False, api_name=False)
    gr.Markdown('[Source code and local app](https://github.com/dilipreddykiralam-png/railway-incident-review) · Hosted inference uses Transformers BF16; evaluate separately from local Ollama results.')

if __name__ == '__main__':
    demo.queue(max_size=8).launch(max_file_size='100mb', show_error=False)
