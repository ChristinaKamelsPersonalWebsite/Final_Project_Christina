import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

import json
import random
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# 🔥 tiny model (fast + small)
MODEL_NAME = "prajjwal1/bert-tiny"
OUTPUT_DIR = "bonus/bert_classifier/model"

LABELS = ["exercise", "nutrition", "program", "progress"]
label2id = {label: i for i, label in enumerate(LABELS)}
id2label = {i: label for label, i in label2id.items()}

EXAMPLES = [
    # exercise
    {"text": "give me chest exercises", "label": "exercise"},
    {"text": "best back workouts", "label": "exercise"},
    {"text": "leg exercises for beginners", "label": "exercise"},
    {"text": "shoulder workout at home", "label": "exercise"},
    {"text": "triceps exercises with dumbbells", "label": "exercise"},
    {"text": "barbell exercises for strength", "label": "exercise"},

    # nutrition
    {"text": "how much protein do i need", "label": "nutrition"},
    {"text": "what should i eat after workout", "label": "nutrition"},
    {"text": "meal plan for fat loss", "label": "nutrition"},
    {"text": "calories needed per day", "label": "nutrition"},
    {"text": "best foods for muscle gain", "label": "nutrition"},
    {"text": "nutrition tips for recovery", "label": "nutrition"},

    # program
    {"text": "build me a workout program", "label": "program"},
    {"text": "create a gym plan", "label": "program"},
    {"text": "weekly workout schedule", "label": "program"},
    {"text": "push pull legs routine", "label": "program"},
    {"text": "4 day training split", "label": "program"},
    {"text": "make me a beginner plan", "label": "program"},

    # progress
    {"text": "track my progress", "label": "progress"},
    {"text": "analyze my workout logs", "label": "progress"},
    {"text": "am i improving", "label": "progress"},
    {"text": "check my strength progress", "label": "progress"},
    {"text": "compare my performance", "label": "progress"},
    {"text": "find plateaus in my training", "label": "progress"},
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": float(acc)}


def manual_split(examples, train_ratio=0.8):
    shuffled = examples[:]
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]


def build_dataset(items):
    return Dataset.from_dict({
        "text": [x["text"] for x in items],
        "label": [label2id[x["label"]] for x in items],
    })


def main():
    set_seed()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    train_examples, val_examples = manual_split(EXAMPLES)

    train_ds = build_dataset(train_examples)
    val_ds = build_dataset(val_examples)

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "labels.json"), "w") as f:
        json.dump({
            "labels": LABELS,
            "metrics": metrics,
        }, f, indent=2)

    print("\n✅ Training complete!")
    print("📁 Model saved to:", OUTPUT_DIR)
    print("📊 Metrics:", metrics)


if __name__ == "__main__":
    main()