# Fit Preview Try-On Service (self-hosted, RunPod Serverless)

This replaces fal.ai with your own GPU-hosted virtual try-on model
([IDM-VTON](https://github.com/yisol/IDM-VTON), open-source, ECCV 2024),
running on [RunPod Serverless](https://runpod.io) — you pay only while a
request is actually processing, and it scales to zero when idle.

## What's here

- `Dockerfile` — clones IDM-VTON, installs its dependencies, bakes in the
  main model weights, and packages it as a RunPod worker.
- `rp_handler.py` — the RunPod entry point. It calls IDM-VTON's own
  `start_tryon` function rather than reimplementing the pipeline, so it
  inherits whatever quality/behavior the upstream repo has.

## Getting the model checkpoints

Good news: the Dockerfile handles this automatically. Both the main
diffusion weights and the human parsing/pose checkpoints are pulled
programmatically from Hugging Face during `docker build` — no manual
downloads needed. (The IDM-VTON README describes the parsing/pose
checkpoints as a manual download; they're actually hosted in the
Hugging Face Space's own repo at a stable path, which is what the
Dockerfile fetches instead.)

## Build and deploy

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/fit-preview-tryon .
docker push YOUR_DOCKERHUB_USERNAME/fit-preview-tryon
```

Then in the RunPod console:

1. Serverless → New Endpoint → Custom Source → point it at your pushed image
2. GPU: pick one with at least 16-24GB VRAM (e.g. RTX 4090 or A5000) —
   IDM-VTON is a diffusion pipeline with multiple sub-models loaded at once
3. Set **Min Workers: 0** (scale to zero, only pay per request) and
   **Max Workers** to whatever concurrency you want
4. Set an **Idle Timeout** (e.g. 5-10s) and a generous **Execution
   Timeout** (diffusion inference can take 20-60s+ per image)
5. Deploy, then grab your **Endpoint ID** and **API Key** from the console

## Test it

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "input": {
          "human_image": "data:image/jpeg;base64,...",
          "garment_image": "data:image/jpeg;base64,...",
          "garment_description": "white cotton t-shirt"
        }
      }'
```

`/runsync` blocks and returns the result directly (good for testing, but
RunPod caps this at ~90s); the web app instead uses `/run` + polling
`/status`, since diffusion inference can run longer than that. See
`lib/tryon.ts` in the web app for that flow.

## A note on output size

RunPod caps handler input/output at 2MB each. A single try-on PNG can
exceed that. If you hit this limit, have the handler upload the result
image to your Supabase `tryon-results` storage bucket directly (using the
service role key, same as the web app does) and return the resulting
public URL instead of the base64 image — `rp_handler.py` returns base64
for now since it's simpler to get running first, but this is the first
thing to change if uploads start failing.

## Realistic expectations

This is meaningfully more work than the fal.ai integration it replaces:
you're now responsible for GPU costs, cold-start latency, checkpoint
management, and keeping up with upstream IDM-VTON changes. It's the right
call if you want full control and no per-image vendor fees at scale; for
just testing the product idea, fal.ai (or a similar hosted API) will get
you there faster. Nothing about this setup has been run end-to-end in
this environment — treat the first real deploy as a debugging session,
not a sure thing.
