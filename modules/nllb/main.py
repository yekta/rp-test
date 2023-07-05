import runpod
from package.constants import (
    DETECTED_CONFIDENCE_SCORE_MIN,
    TARGET_LANG_FLORES,
    TARGET_LANG_SCORE_MAX,
)
from package.translate import translate_text
from package.setup import setup
import time

pack = setup()


def handler(event):
    print(event)

    s = time.time()
    print("💬 Translating text...")
    text_1 = event["input"]["text_1"]
    text_2 = event["input"]["text_2"]
    translated_text_1 = translate_text(
        text=text_1,
        text_flores=None,
        detector=pack.detector,
        model=pack.translator,
        tokenizer=pack.tokenizer,
        target_flores=TARGET_LANG_FLORES,
        target_score_max=TARGET_LANG_SCORE_MAX,
        detected_confidence_score_min=DETECTED_CONFIDENCE_SCORE_MIN,
        label="Text 1",
    )
    translated_text_2 = translate_text(
        text=text_2,
        text_flores=None,
        detector=pack.detector,
        model=pack.translator,
        tokenizer=pack.tokenizer,
        target_flores=TARGET_LANG_FLORES,
        target_score_max=TARGET_LANG_SCORE_MAX,
        detected_confidence_score_min=DETECTED_CONFIDENCE_SCORE_MIN,
        label="Text 2",
    )

    e = time.time()
    print(f"✅ Translated text in: {round(e-s, 2)} seconds")

    return f"{translated_text_1} | {translated_text_2}"


runpod.serverless.start({"handler": handler})
