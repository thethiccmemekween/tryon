"""
RunPod serverless handler for IDM-VTON.

Rather than reimplementing IDM-VTON's pipeline (mask generation, pose
detection, the actual diffusion inference), this imports and calls the
function the official gradio_demo/app.py already defines and has verified
to work end-to-end. As of writing, that function is `start_tryon` and takes
roughly: a human image (as a dict with a "background" key, matching
Gradio's ImageEditor component), a garment image, a text description of
the garment, and some booleans/ints for masking and sampling options.

IMPORTANT: verify this against your actual checked-out copy of
gradio_demo/app.py before relying on it — upstream repos change, and this
was written from public documentation rather than a live test run (this
environment has no GPU to execute it against). If the import or call
signature below doesn't match, open gradio_demo/app.py in your container
and adjust the two spots marked ADAPT below.
"""

import base64
import io
import os
import sys

import runpod
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.insert(0, '/workspace/IDM-VTON')
sys.path.insert(0, '/workspace/IDM-VTON/gradio_demo')

# The container disk on this endpoint is too small to hold the ~17GB of
# weights (that's what caused the earlier "No space left on device" crash),
# so weights are downloaded onto the attached RunPod Network Volume
# (mounted at /runpod-volume) instead of the container's own disk. IDM-VTON's
# code expects checkpoints at fixed paths under the repo directory
# (IDM-VTON/ckpt_hf, IDM-VTON/ckpt), so we symlink those paths to the volume
# rather than changing IDM-VTON's own code. This also means only the first
# worker ever downloads anything — later workers reuse what's already on
# the volume.
if not os.path.isdir('/runpod-volume'):
    raise RuntimeError(
        'Expected a RunPod Network Volume mounted at /runpod-volume — attach '
        'one to this endpoint (Edit Endpoint > Network volumes). The '
        'container disk alone is not large enough for IDM-VTON\'s weights.'
    )

WEIGHTS_ROOT = '/runpod-volume/idm-vton-weights'
os.makedirs(WEIGHTS_ROOT, exist_ok=True)

MAIN_WEIGHTS_DIR = os.path.join(WEIGHTS_ROOT, 'ckpt_hf')
print('Downloading main diffusion weights...')
snapshot_download(repo_id='yisol/IDM-VTON', local_dir=MAIN_WEIGHTS_DIR)

print('Downloading human parsing / pose checkpoints...')
snapshot_download(
    repo_id='yisol/IDM-VTON',
    repo_type='space',
    allow_patterns=['ckpt/*'],
    local_dir=WEIGHTS_ROOT
)
print('Model weights ready.')

for target_name in ('ckpt_hf', 'ckpt'):
    link_path = f'/workspace/IDM-VTON/{target_name}'
    volume_path = os.path.join(WEIGHTS_ROOT, target_name)
    if not os.path.lexists(link_path):
        os.symlink(volume_path, link_path)

# ADAPT #1: import path/function name — check gradio_demo/app.py in your
# checkout if this fails.
from app import start_tryon  # noqa: E402


def _decode_image(data: str) -> Image.Image:
    """Accepts either a data: URI or raw base64."""
    if data.startswith('data:'):
        data = data.split(',', 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(data))).convert('RGB')


def _encode_image(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')


def handler(job):
    job_input = job['input']

    try:
        human_img = _decode_image(job_input['human_image'])
        garment_img = _decode_image(job_input['garment_image'])
        garment_description = job_input.get('garment_description', 'a clothing item')
        use_auto_mask = job_input.get('use_auto_mask', True)
        use_auto_crop = job_input.get('use_auto_crop', False)
        denoise_steps = job_input.get('denoise_steps', 30)
        seed = job_input.get('seed', 42)

        # ADAPT #2: match the real start_tryon(...) signature in your checkout.
        # The Gradio ImageEditor component normally passes a dict; a plain
        # PIL image is used here as the common case for a single try-on
        # (no manual mask drawing) — adjust if your checkout expects the
        # dict form instead.
        result_img, _mask_img = start_tryon(
            {'background': human_img, 'layers': [], 'composite': None},
            garment_img,
            garment_description,
            use_auto_mask,
            use_auto_crop,
            denoise_steps,
            seed,
        )

        return {'image': _encode_image(result_img)}

    except Exception as e:
        return {'error': str(e)}


runpod.serverless.start({'handler': handler})
