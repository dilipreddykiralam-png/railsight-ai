# Hosting and maintenance

## Public application

The live [RailSight AI Space](https://huggingface.co/spaces/dilipbobby/railsight-ai) runs Qwen2.5-VL 7B with Transformers on Hugging Face ZeroGPU. The Space is free to try, but requests can queue and each visitor has a daily GPU allowance. Video analysis makes several model calls, so it is more likely to stop partway when quota runs out. A partial video run is shown as partial and is not counted in the public total.

The hosted model and software differ from local Ollama runs. Record the deployment and model settings shown in each report, and evaluate the two configurations separately. The app is a research prototype, not a railway safety system.

## Public completed-analysis counter

The hosted app displays **Images analyzed**, **Videos analyzed**, and **Total completed analyses**. The README badge displays the same combined total. One image counts once; one video counts once if all sampled frames finish. Failed or partial analyses, individual video frames, human verification and demo-mode examples are excluded. Reanalyzing an upload creates a new run and counts again if successful. These are completed runs, not unique files, page views, unique people, or active users. Local Ollama runs are separate and do not change the hosted count.

The small public Dataset repository `dilipbobby/railsight-ai-usage` stores aggregate totals, a badge label, and short-lived hashes of random report IDs to avoid duplicate increments. It stores no image/video, model response, filename, reviewer detail, IP address, or account identity. The public JSON is intentionally visible so GitHub can display the badge. The app can read the public totals without a token; to increment them, add a fine-grained Hugging Face write token restricted to this one Dataset repository as the Space secret `RAILSIGHT_USAGE_TOKEN`. Never put the token in Git or a public Space variable. Set the optional non-sensitive Space variable `RAILSIGHT_USAGE_REPO` only if the counter repo name changes. Existing totals migrate automatically; earlier runs without media types are preserved separately rather than assigned invented image/video labels.

To connect or repair the count:

1. Check the existing [counter Dataset](https://huggingface.co/datasets/dilipbobby/railsight-ai-usage/tree/main). **Do not reset or overwrite existing totals.** Only if creating a fresh Dataset, add `usage.json` with this initial content:

   ```json
   {"schemaVersion":1,"label":"completed analyses","message":"0","color":"blue","cacheSeconds":300,"counts":{"images":0,"videos":0,"legacy":0},"event_keys":[]}
   ```

2. Open [Access Tokens](https://huggingface.co/settings/tokens) → **Create new token** → **Fine-grained**. Name it `railsight-usage`. Under permissions for specific repositories, select **dilipbobby/railsight-ai-usage** and enable reading and writing that repository's contents. No account-wide write permission is needed.
3. Copy the token directly into [Space Settings](https://huggingface.co/spaces/dilipbobby/railsight-ai/settings) → **Variables and secrets → New secret**. Name: `RAILSIGHT_USAGE_TOKEN`. Value: your token. Save it. Do not paste the token into a chat, README, or public file.
4. Deploy the current files using the instructions below. If the app has not restarted after adding the secret, restart it from Space Settings. Complete a successful image analysis and check that **Images analyzed** increases by one. A complete video run increases **Videos analyzed** by one, regardless of sampled-frame count. Check the saved `usage.json` and README badge; the badge may take several minutes to refresh because it is cached.

The counter only records runs after write access is configured. Earlier page visits or ZeroGPU calls cannot be converted into completed image/video counts: one video or validation retry can use several GPU calls. Never backfill those analytics as completed analyses.

The app shows **Counter not connected** when the secret is missing, and **Counter update failed** if saving a completed run fails. A failed write is not reported as success; its report has `usage_count_saved: false`. Analysis and downloads remain available. Totals refresh automatically and can be refreshed with the app's button. This small prototype counter does not recover unsaved runs after a server restart. Rotate or revoke the token in Hugging Face settings if it is exposed. Hugging Face provides [Space secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables), [fine-grained access tokens](https://huggingface.co/docs/hub/security-tokens), and [public Dataset storage](https://huggingface.co/docs/hub/storage-limits).

## Updating the Hugging Face app

The maintained source is `hosting/huggingface/`. From the repository root, prepare the four Space files:

```bash
python scripts/build_hf_space.py
```

Upload `app.py`, `requirements.txt`, `README.md`, and `railreview.zip` from `dist/huggingface/` to the Space and commit them. The archive contains the public application modules and prompts, not user media or private research data. Keep the Space on **ZeroGPU Free**. After a code or dependency change, check an image, a video, human correction, download, and quota/partial-failure behavior.

The hosted app keeps uploads and downloaded reports in temporary storage and removes them during cleanup. Users should download reports before leaving. Hugging Face's [Space storage documentation](https://huggingface.co/docs/hub/spaces-storage) explains that the default Space disk is temporary; the counter therefore uses its separate Dataset repository.

Deployment checks on 27 September 2026 confirmed a licensed image analysis and saved human correction. A two-frame synthetic video (the same still repeated) produced one valid frame and one ZeroGPU quota error. This checks partial-failure handling only; it is not a research example or a claim of unrestricted video availability.
