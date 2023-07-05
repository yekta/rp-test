import torch
from diffusers import (
    DiffusionPipeline,
)
import os
from huggingface_hub import _login

MODEL_CACHE = "MODELS_CACHE"

hf_token = os.environ.get("HUGGINGFACE_TOKEN", None)
if hf_token is not None:
    print(f"⏳ Logging in to HuggingFace")
    _login.login(token=hf_token)
    print(f"✅ Logged in to HuggingFace")

t2i_pipe = DiffusionPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion",
    torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE,
)
