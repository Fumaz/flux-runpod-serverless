import base64
import os
from io import BytesIO

import runpod
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(os.environ.get("FLUX_MODEL_NAME", "black-forest-labs/FLUX.1-dev"),
                                    torch_dtype=torch.bfloat16, cache_dir=os.environ.get("CACHE_DIR", "/runpod-volume/models"))


def handler(job):
    job_input = job["input"]
    prompt = job_input["prompt"]
    height = job_input["height"]
    width = job_input["width"]
    guidance_scale = job_input["guidance_scale"]
    num_inference_steps = job_input["num_inference_steps"]

    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512
    ).images[0]

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return {"output": img_str}


runpod.serverless.start({"handler": handler})
