# Hosting and maintenance

## Public application

The live [RailSight AI Space](https://huggingface.co/spaces/dilipbobby/railsight-ai) runs Qwen2.5-VL 7B with Transformers on Hugging Face ZeroGPU. The Space is free to try, but requests can queue and each visitor has a daily GPU allowance. Video analysis makes several model calls, so it is more likely to stop partway when quota runs out. A partial video run is shown as partial and is not counted in the public total.

The hosted model and software differ from local Ollama runs. Record the deployment and model settings shown in each report, and evaluate the two configurations separately. The app is a research prototype, not a railway safety system.

## Public completed-analysis counter

The README badge and hosted app display a shared count of **successful, complete VLM analyses**. One image counts once; one video counts once if all sampled frames finish. Failed or partial analyses and demo-mode examples are excluded. This is a count of completed runs, not page views, unique people, or active users.

The small public Dataset repository `dilipbobby/railsight-ai-usage` stores only the total, a badge label, and short-lived hashes of random report IDs to avoid duplicate increments. It stores no image/video, model response, filename, reviewer detail, IP address, or account identity. The public JSON is intentionally visible so GitHub can display the badge. The app needs a fine-grained Hugging Face write token restricted to this one Dataset repository, stored as the Space secret `RAILSIGHT_USAGE_TOKEN`; never put the token in Git. Set the optional non-sensitive Space variable `RAILSIGHT_USAGE_REPO` only if the counter repo name changes.

To connect or repair the count:

1. Create a **public Dataset** named `railsight-ai-usage` under the `dilipbobby` Hugging Face account. Add `usage.json` with this initial content:

   ```json
   {"schemaVersion":1,"label":"completed analyses","message":"0","color":"blue","cacheSeconds":300,"event_keys":[]}
   ```

2. Create a Hugging Face **fine-grained** token that can write only to this Dataset repository. Add it to the Space under **Settings → Variables and secrets** as the secret `RAILSIGHT_USAGE_TOKEN`.
3. Deploy the current files from `hosting/huggingface/` to the Space. The counter starts from zero when first enabled; it cannot reconstruct earlier uses.
4. Open the Space, complete a successful image analysis, then check `usage.json` and the README badge. The badge may take several minutes to refresh because it is cached.

If the counter is unavailable, image/video analysis still works; the number may remain unchanged until the write succeeds. Rotate or revoke the token in Hugging Face settings if it is exposed. Hugging Face provides [Space secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables), [fine-grained access tokens](https://huggingface.co/docs/hub/security-tokens), and [public Dataset storage](https://huggingface.co/docs/hub/storage-limits).

## Updating the Hugging Face app

The maintained source is `hosting/huggingface/`. From the repository root, prepare the four Space files:

```bash
python scripts/build_hf_space.py
```

Upload `app.py`, `requirements.txt`, `README.md`, and `railreview.zip` from `dist/huggingface/` to the Space and commit them. The archive contains the public application modules and prompts, not user media or private research data. Keep the Space on **ZeroGPU Free**. After a code or dependency change, check an image, a video, human correction, download, and quota/partial-failure behavior.

The hosted app keeps uploads and downloaded reports in temporary storage and removes them during cleanup. Users should download reports before leaving. Hugging Face's [Space storage documentation](https://huggingface.co/docs/hub/spaces-storage) explains that the default Space disk is temporary; the counter therefore uses its separate Dataset repository.

Deployment checks on 27 September 2026 confirmed a licensed image analysis and saved human correction. A two-frame synthetic video (the same still repeated) produced one valid frame and one ZeroGPU quota error. This checks partial-failure handling only; it is not a research example or a claim of unrestricted video availability.
