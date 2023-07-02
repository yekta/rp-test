from typing import Any
from diffusers import (
    DiffusionPipeline,
)
import torch
from constants import MODEL_CACHE


class ModelsPack:
    def __init__(
        self,
        kandinsky: Any,
    ):
        self.kandinsky = kandinsky


def setup() -> ModelsPack:
    # Kandinsky
    print("⏳ Loading Kandinsky")
    pipe_prior = DiffusionPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-1-prior",
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE,
    )
    pipe_prior.to("cuda")

    t2i_pipe = DiffusionPipeline.from_pretrained(
        "kandinsky-community/kandinsky-2-1",
        torch_dtype=torch.float16,
        cache_dir=MODEL_CACHE,
    )
    t2i_pipe.to("cuda")
    kandinsky = {
        "prior": pipe_prior,
        "text2img": t2i_pipe,
    }
    print("✅ Loaded Kandinsky")
    return ModelsPack(
        kandinsky=kandinsky,
    )
