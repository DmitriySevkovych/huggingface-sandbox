import os
from typing import List

from transformers import pipeline


def test_pipeline():
    classifier = pipeline("image-classification")
    preds = classifier(images=[os.path.join(os.getcwd(), "data", "xray.jpg")])
    return [
        {"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds[0]
    ]
