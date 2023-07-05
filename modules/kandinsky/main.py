import runpod
from package.setup import setup
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

    kandinsky = pack.kandinsky
    prior_pipe = kandinsky["prior"]
    t2i_pipe = kandinsky["text2img"]

    image_embeds, negative_image_embeds = prior_pipe(
        prompt, guidance_scale=1.0
    ).to_tuple()

    image = t2i_pipe(
        prompt,
        image_embeds=image_embeds,
        negative_image_embeds=negative_image_embeds,
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
