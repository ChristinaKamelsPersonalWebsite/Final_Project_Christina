import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

MODEL_DIR = "bonus/bert_classifier/model"

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model

    if _tokenizer is None:
        _tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)

    if _model is None:
        _model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()

    return _tokenizer, _model


@torch.no_grad()
def classify_query(text: str):
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=64,
    )

    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)[0]

    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())
    label = model.config.id2label[pred_id]

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "scores": {
            model.config.id2label[i]: round(float(probs[i].item()), 4)
            for i in range(len(probs))
        },
    }


if __name__ == "__main__":
    test_queries = [
        "give me chest exercises",
        "how much protein should i eat",
        "make me a 4 day workout plan",
        "analyze my progress logs",
    ]

    for q in test_queries:
        print(q, "->", classify_query(q))