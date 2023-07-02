import runpod
from models.setup import setup
import time

models_pack = setup()


def handler(event):
    print(event)

    s = time.time()
    print("🎨 Generating image")
    prompt = event["input"]["prompt"]
    kandinsky = models_pack.kandinsky
    prior_pipe = kandinsky["prior"]
    t2i_pipe = kandinsky["text2img"]

    image_embeds, negative_image_embeds = prior_pipe(
        prompt, guidance_scale=1.0
    ).to_tuple()

    image = t2i_pipe(
        prompt,
        image_embeds=image_embeds,
        negative_image_embeds=negative_image_embeds,
        height=768,
        width=768,
    ).images[0]
    e = time.time()
    print(f"✅ Generated image in: {round(e-s, 2)} seconds")

    return image


runpod.serverless.start({"handler": handler})
