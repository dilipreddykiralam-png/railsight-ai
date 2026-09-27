---
title: RailSight AI
emoji: 🚆
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
python_version: '3.10'
app_file: app.py
pinned: false
models:
  - Qwen/Qwen2.5-VL-7B-Instruct
short_description: Railway damage analysis with Qwen 7B and human review
---
# RailSight AI
Railway Incident & Damage Intelligence.

Upload a railway image or short video, inspect provisional findings and supporting frames, correct the structured result, and download your reviewed report.

Runs Qwen2.5-VL 7B on Hugging Face ZeroGPU with Transformers (BF16), not the local Ollama quantized backend. Hosted results are a separate inference configuration and should not be mixed with the local evaluation without recording this difference. Video analysis samples frames independently; it does not analyze motion or every moment of the clip. Confidence is an uncalibrated model estimate, not measured accuracy.

Media and reports use temporary server storage and session state, with periodic cache cleanup. Do not upload sensitive media. Reports are not committed to this repository and do not train the model. Download before leaving. The host's privacy policy also applies. When the usage counter is enabled, it records only successful complete analyses: one image or one full video counts once. The number is not a count of unique people. Quotas and queues apply.

Source and local installation: https://github.com/dilipreddykiralam-png/railsight-ai

The `railreview.zip` file is generated from the public repository's Python package and prompts by `scripts/build_hf_space.py`; it contains no private datasets. Model and dependency licenses are separate. This Space does not add a project-wide license grant.
