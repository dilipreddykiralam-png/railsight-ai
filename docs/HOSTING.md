# Sharing the application for free

Checked against official documentation on 25 September 2026. Provider eligibility and quotas can change.

## What is available now

- **GitHub repository:** full runnable Streamlit application and reproducible examples.
- **GitHub Pages sample demo:** three licensed images, actual saved Qwen 7B responses, and downloadable review notes. This is a static showcase, not live model inference.
- **Local full application:** upload images/videos and run Qwen2.5-VL 7B using Ollama without a hosted inference subscription. Your computer supplies the compute.
- **Public live inference:** [RailSight AI on Hugging Face](https://huggingface.co/spaces/dilipbobby/railsight-ai), running on ZeroGPU. Runtime session/test counts reset on restart; no all-time public count is claimed.

[GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages) hosts static websites, not Python/Ollama servers. GitHub Actions runs software checks and publishes the sample website; it is not used as a public inference server.

## Best free live-demo candidate: Hugging Face ZeroGPU

The current [ZeroGPU documentation](https://huggingface.co/docs/hub/spaces-zerogpu) allows eligible personal free accounts (verified email, older than 30 days, in good standing) to host up to two ZeroGPU Spaces. It supports **Gradio**, so the existing Streamlit/Ollama interface cannot be uploaded unchanged. The deployed adapter reuses the schema, prompts and review logic in a Gradio interface and runs Qwen2.5-VL 7B through Transformers.

The Gradio adaptation is deployed. A real image, report download and human correction were checked on the cloud service; these are deployment checks, not recognition accuracy measurements. It would use different inference software and potentially different quantization from the local Ollama run; its outputs must be evaluated separately. Begin with single-image uploads; video sampling consumes substantially more GPU time. Free users face queues and daily GPU quotas. No unlimited availability is promised.

The general [Spaces overview](https://huggingface.co/docs/hub/spaces-overview) says ordinary new Docker/Gradio compute Spaces require a paid plan, even though CPU Basic has no hourly hardware charge. The ZeroGPU free-account exception is described in its dedicated documentation. Check the account's actual creation options before deploying; do not select paid hardware.

Steps for the owner:

1. Sign in to Hugging Face and verify your email.
2. Check account age and whether ZeroGPU creation is available.
3. Adapt the UI and inference backend to Gradio + Transformers, keeping the local app unchanged.
4. Test a normal scene and an incident, validation failure handling, review downloads and GPU quota errors.
5. Publish the Space link only once live inference is verified. Add a link at the top of the GitHub README.

[Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud) can host a free Python front end, but that does not supply a Qwen 7B GPU server. An external inference endpoint would still be needed. It is not a complete free replacement by itself.

## Optional real usage counter (implemented for the Python app)

Tracking is **off by default**. A public deployment can set:

```bash
export RAILREVIEW_PUBLIC=1
export RAILREVIEW_USAGE_DB=/persistent-volume/usage.sqlite3
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

PowerShell uses `$env:RAILREVIEW_PUBLIC = "1"` and `$env:RAILREVIEW_USAGE_DB = "C:\persistent-volume\usage.sqlite3"` instead. The path must refer to writable, persistent storage. Public mode skips permanent review-file saving; visitors download their reports from their session. Inference-provider and hosting-provider data handling still apply.

- **Completed tests:** successful VLM analyses; uncertain but schema-valid predictions count. Demo mode, failures and partially failed videos do not count. One complete video is one test, not one test per frame.
- **Testing sessions:** anonymous Streamlit sessions that completed at least one test. Reloading/reconnecting may create a new session. These are **not unique people**, geographic reach, accuracy, or currently active users. Bots/repeated manual tests can count.
- The database holds only random session IDs and analysis IDs; no media, filenames, reviewer names, IP addresses or country data. Duplicate analysis IDs cannot increment totals twice. Local runs are excluded unless deliberately configured as public.
- Database outages do not prevent analysis; missed events are not retroactively recovered. Back up the database. A lost database resets counts.
- The SQLite implementation is for **one host with a shared persistent disk**, not a distributed deployment. Do not call counts all-time on ephemeral storage. Hugging Face's [default Space disk is ephemeral](https://huggingface.co/docs/hub/spaces-storage); durable storage must be resolved before enabling an all-time badge there.

For a future server deployment, start `python -m railreview.usage_server` with the same database environment. It provides read-only `/usage.json` and `/badge.json` on port 8502. Put it behind your HTTPS reverse proxy. Never expose the Ollama port directly. The badge endpoint cannot increment counts. The optional counter is not wired into the static demo or a future Gradio adapter.

After obtaining an actual deployed HTTPS badge URL, add the following near the README title (replace the placeholder with the URL-encoded endpoint):

```markdown
![Testing sessions](https://img.shields.io/endpoint?url=ENCODED_HTTPS_BADGE_URL)
```

[Shields endpoint badges](https://shields.io/badges/endpoint-badge) are cached, so this is an aggregate count refreshed periodically, not an instantaneous live-presence indicator. Until a backend is deployed, the README shows **not connected** rather than a made-up count.

## Maintain the deployed Space

The maintained source is `hosting/huggingface/`. Build its four upload files with:

```bash
python scripts/build_hf_space.py
```

In the Space **Files → Contribute → Upload files**, upload `app.py`, `requirements.txt`, `README.md`, and `railreview.zip` from `dist/huggingface/` together, then commit. The archive contains only public Python modules and prompts; the build does not include user uploads or datasets. Keep hardware set to **ZeroGPU Free**. No API token is required for this browser workflow.

The hosted model revision and generation settings are recorded in downloaded reports. `scripts/check_hf_ui.py` checks the Gradio interface and callbacks with fixture inference in an environment containing Gradio 5.49.1 and the app's lightweight dependencies; it does not load a GPU model. Run the real deployment smoke checks after dependency or inference changes.

The hosted app limits video sampling to 2–4 frames to conserve free GPU quota. It retains all original frame responses in the report. Image/video uploads and review downloads use temporary storage with cleanup. Runtime counters retain only aggregate totals; a session flag prevents repeated tests in one session from increasing the session count. No IP or country tracking is implemented. The initial deployment checks also count as completed tests, so totals must not be described as independent users or research cases.

Deployment check on 27 September 2026: a licensed image completed real Qwen inference in 12.123 seconds, and a corrected asset list was saved and downloaded with the original retained. A two-frame synthetic video (one still image repeated, not a research example) produced one valid frame and one ZeroGPU runs-limit error. This verifies partial-failure handling, not unrestricted complete-video availability. Sign in on the Space page for the account quota; free usage remains limited.
