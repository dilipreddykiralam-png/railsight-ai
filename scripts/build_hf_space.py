"""Create a small browser-uploadable Space distribution, excluding all user media."""
from pathlib import Path
import shutil
import zipfile
ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'dist' / 'huggingface'
out.mkdir(parents=True, exist_ok=True)
for name in ('app.py', 'requirements.txt', 'README.md'):
    shutil.copy(ROOT / 'hosting/huggingface' / name, out / name)
with zipfile.ZipFile(out / 'railreview.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
    for pattern in ('railreview/*.py', 'prompts/*.txt', 'prompts/*.json'):
        for path in sorted(ROOT.glob(pattern)):
            bundle.write(path, str(path.relative_to(ROOT)))
print(out)
