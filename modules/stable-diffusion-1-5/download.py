import torch
from diffusers import (
    DiffusionPipeline,
)

MODEL_CACHE = "MODELS_CACHE"

t2i_pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE,
)
