import runpod
from .setup import setup
import time

pack = setup()


def handler(event):
    print(event)

    s = time.time()
    print("🎨 Generating image")
    prompt = event["input"]["prompt"]
    width = event["input"]["width"]
    height = event["input"]["height"]
    num_inference_steps = event["input"]["num_inference_steps"]
    num_images_per_prompt = event["input"]["num_images_per_prompt"]
    guidance_scale = event["input"]["guidance_scale"]
    pipe = pack.pipe

    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=guidance_scale,
    ).images[0]
    e = time.time()
    print(f"✅ Generated image in: {round(e-s, 2)} seconds")

    return "Done"


runpod.serverless.start({"handler": handler})
