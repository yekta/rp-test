import torch
from diffusers import (
    DiffusionPipeline,
)
from constants import MODEL_CACHE

t2i_pipe = DiffusionPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1",
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE,
)

pipe_prior = DiffusionPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-1-prior",
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE,
)
