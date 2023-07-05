from typing import Any
from lingua import LanguageDetectorBuilder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .constants import (
    TRANSLATOR_MODEL_CACHE,
    TRANSLATOR_TOKENIZER_CACHE,
    TRANSLATOR_MODEL_ID,
)


class Pack:
    def __init__(
        self,
        translator: Any,
        tokenizer: Any,
        detector: Any,
    ):
        self.translator = translator
        self.tokenizer = tokenizer
        self.detector = detector


def setup() -> Pack:
    """Load the model into memory to make running multiple predictions efficient"""
    print("Loading language detector...")
    detector = (
        LanguageDetectorBuilder.from_all_languages()
        .with_preloaded_language_models()
        .build()
    )
    print("Loaded language detector!")

    print("Loading translator...")
    tokenizer = AutoTokenizer.from_pretrained(
        TRANSLATOR_MODEL_ID, cache_dir=TRANSLATOR_TOKENIZER_CACHE
    )
    translator = AutoModelForSeq2SeqLM.from_pretrained(
        TRANSLATOR_MODEL_ID, cache_dir=TRANSLATOR_MODEL_CACHE
    )
    print("Loaded translator!")
    return Pack(translator=translator, tokenizer=tokenizer, detector=detector)
