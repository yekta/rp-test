from typing import Any
from diffusers import (
    DiffusionPipeline,
)
import torch
from .constants import MODEL_CACHE


class Pack:
    def __init__(
        self,
        pipe: Any,
    ):
        self.pipe = pipe


def setup() -> Pack:
    print("⏳ Loading 22h Diffusion")
    t2i_pipe = DiffusionPipeline.from_pretrained(
        "22h/vintedois-diffusion-v0-1",
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE,
    )
    t2i_pipe.to("cuda")
    print("✅ Loaded 22h Diffusion")
    return Pack(
        pipe=t2i_pipe,
    )
