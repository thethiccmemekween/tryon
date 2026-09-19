# RunPod serverless worker for IDM-VTON virtual try-on.
# Base image includes CUDA + PyTorch already set up for GPU inference.
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Clone the official IDM-VTON repo and use its code as-is ---
# Pinning a commit (rather than floating on `main`) is strongly recommended
# once you've confirmed a version that works for you, so a future upstream
# change can't silently break your worker. Replace <COMMIT_SHA> below.
RUN git clone https://github.com/yisol/IDM-VTON.git /workspace/IDM-VTON
# RUN cd /workspace/IDM-VTON && git checkout <COMMIT_SHA>

WORKDIR /workspace/IDM-VTON

# --- Python deps: the repo's own requirements, plus the RunPod SDK ---
# environment.yaml is a conda file; pull just the pip dependencies out of it
# if `pip install -r requirements.txt` isn't present/sufficient in your
# checkout — check the repo for the current install instructions.
RUN pip install --no-cache-dir -r requirements.txt || true
RUN pip install --no-cache-dir runpod huggingface_hub pillow

# --- Model weights are NOT downloaded here ---
# They're downloaded at container startup instead (see rp_handler.py),
# because fetching ~17GB during the build exceeds RunPod's 30-minute
# GitHub build timeout. This keeps the build itself fast; the tradeoff is
# a slower first request after each cold start (see rp_handler.py's
# comments, and the README, for how to avoid repeating this on every
# cold start using a RunPod Network Volume).

COPY rp_handler.py /workspace/IDM-VTON/rp_handler.py

CMD ["python3", "-u", "rp_handler.py"]
