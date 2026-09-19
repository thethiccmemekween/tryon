# Fit Preview Try-On Service (self-hosted, RunPod Serverless)

This replaces fal.ai with your own GPU-hosted virtual try-on model
([IDM-VTON](https://github.com/yisol/IDM-VTON), open-source, ECCV 2024),
running on [RunPod Serverless](https://runpod.io) — you pay only while a
request is actually processing, and it scales to zero when idle.

## What's here

- `Dockerfile` — clones IDM-VTON, installs its dependencies, and packages
  it as a RunPod worker. Deliberately does NOT download model weights at
  build time — see below.
- `rp_handler.py` — the RunPod entry point. Downloads model weights on
  startup, then calls IDM-VTON's own `start_tryon` function rather than
  reimplementing the pipeline, so it inherits whatever quality/behavior
  the upstream repo has.

## A build timeout you will hit if weights download during the build

RunPod's GitHub-based builds have a 30-minute limit. Downloading the
model weights (~17GB+) during the build easily exceeds that — this is
why the Dockerfile does not do it. Instead, `rp_handler.py` downloads
everything itself the moment a worker container starts up.

The tradeoff: the **first request after each cold start** will be slow
(several minutes, while it downloads everything) rather than instant.
Subsequent requests to that same warm worker are fast as normal.

**A RunPod Network Volume is required, not optional**, on this endpoint —
the container's own disk isn't big enough to hold the ~17GB of weights
(this showed up as a `No space left on device` crash-loop in worker logs
the first time this was deployed). `rp_handler.py` downloads weights onto
`/runpod-volume/idm-vton-weights` and symlinks them into place at
`IDM-VTON/ckpt_hf` and `IDM-VTON/ckpt` so the upstream code finds them at
its expected paths. Attach a volume (Edit Endpoint > Network volumes,
same datacenter region as your GPU workers, 50GB+) before deploying. As
a side benefit, only the first worker ever downloads anything — later
workers reuse what's already on the volume instead
of re-downloading.

## Getting the model checkpoints

Both the main diffusion weights and the human parsing/pose checkpoints
are pulled programmatically from Hugging Face — no manual downloads
needed. (The IDM-VTON README describes the parsing/pose checkpoints as a
manual download; they're actually hosted in the Hugging Face Space's own
repo at a stable path, which is what `rp_handler.py` fetches instead.)

## Deploy

Two ways to get this running on RunPod:

**From GitHub (recommended if your local machine has limited disk space)**
— RunPod builds the image on their own servers, so your computer's disk
never comes into play:
1. Push this folder to a GitHub repo
2. RunPod console → Settings → Connections → connect GitHub
3. Serverless → New Endpoint → GitHub Repo → select the repo
4. GPU: pick one with at least 16-24GB VRAM (e.g. RTX 4090 or A5000) —
   IDM-VTON is a diffusion pipeline with multiple sub-models loaded at once
5. Set **Min Workers: 0** and a generous **Execution Timeout** (cold
   starts that include the weight download can take several minutes;
   normal inference is faster but still 20-60s+ per image)
6. Deploy, then grab your **Endpoint ID** and **API Key** from the console

**From Docker Hub (if you'd rather build locally)**
```bash
docker build -t YOUR_DOCKERHUB_USERNAME/fit-preview-tryon .
docker push YOUR_DOCKERHUB_USERNAME/fit-preview-tryon
```
Then Serverless → New Endpoint → Custom Source → point it at the pushed
image, and configure GPU/workers/timeout as above.

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

The first call after a cold start will take a while (downloading
weights) — don't assume it's broken if it's slow the first time.
`/runsync` blocks and returns the result directly (good for testing, but
RunPod caps this at ~90s, which the weight-download cold start will
exceed); the web app instead uses `/run` + polling `/status`, since
inference (and especially a cold-start download) can run longer than
that. See `lib/tryon.ts` in the web app for that flow.

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
