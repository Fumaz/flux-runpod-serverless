import base64
import os
from io import BytesIO

import runpod
import torch
from PIL import Image
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(os.environ.get("FLUX_MODEL_NAME", "black-forest-labs/FLUX.1-dev"),
                                    torch_dtype=torch.float16,
                                    cache_dir=os.environ.get("CACHE_DIR", "/runpod-volume/models"))

pipe = pipe.to("cuda")


def handler(job):
    job_input = job["input"]
    prompt = job_input["prompt"]
    guidance_scale = job_input["guidance_scale"]
    num_inference_steps = job_input["num_inference_steps"]

    init_image_str = job_input.get("init_image")

    if init_image_str:
        init_image = Image.open(BytesIO(base64.b64decode(init_image_str)))
        aspect_ratio = init_image.width / init_image.height

        max_size = job_input.get("max_size", 2048)
        if aspect_ratio > 1:
            width = min(init_image.width, max_size)
            height = int(width / aspect_ratio)
        else:
            height = min(init_image.height, max_size)
            width = int(height * aspect_ratio)

        init_image = init_image.resize((width, height))
    else:
        width = job_input.get("width", 1024)
        height = job_input.get("height", 1024)
        init_image = None

    image = pipe(
        prompt=prompt,
        image=init_image,
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
