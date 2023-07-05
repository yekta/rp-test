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
    print("⏳ Loading Luna Diffusion v1.5")
    t2i_pipe = DiffusionPipeline.from_pretrained(
        "proximasanfinetuning/luna-diffusion",
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE,
    )
    t2i_pipe.to("cuda")
    print("✅ Loaded Luna Diffusion v1.5")
    return Pack(
        pipe=t2i_pipe,
    )
