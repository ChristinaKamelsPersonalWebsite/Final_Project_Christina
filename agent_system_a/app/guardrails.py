from __future__ import annotations

from typing import Tuple
import re

BLOCKED_PATTERNS = [
    "steroid",
    "steroids",
    "anabolic",
    "tren",
    "testosterone cycle",
    "ped",
    "performance enhancing drug",
    "starve",
    "stop eating",
    "eat 500 calories",
    "extreme cut",
    "purge",
    "vomit after eating",
    "laxative",
    "diagnose my injury",
    "medical diagnosis",
    "treat my fracture",
    "how to hide weight loss",
    "drugs to lose weight",
    "weight loss drugs",
    "diet pills",
    "fat burner pills",
]

HEALTH_SENSITIVE_PATTERNS = [
    "injury",
    "pain",
    "doctor",
    "medical",
    "diagnosis",
    "eating disorder",
    "fracture",
    "sprain",
    "tear",
]

EXTREME_TRAINING_PATTERNS = [
    "train 7 days no rest",
    "work out all day",
    "never rest",
    "overtrain",
]


def validate_input(message: str) -> Tuple[bool, str]:
    text = message.lower().strip()

    
    for pattern in BLOCKED_PATTERNS:
            if re.search(r'\b' + re.escape(pattern) + r'\b', text):
                return (
                False,
                "I can help with general fitness and nutrition guidance, but I can’t help with dangerous, drug-related, or medical-treatment requests.",
            )

    for pattern in EXTREME_TRAINING_PATTERNS:
        if pattern in text:
            return (
                False,
                "I can help you build an effective training plan, but not one that encourages unsafe overtraining or harmful behavior.",
            )

    return True, ""


def needs_safety_note(message: str) -> bool:
    text = message.lower().strip()
    return any(pattern in text for pattern in HEALTH_SENSITIVE_PATTERNS)


def sanitize_output(response: str) -> str:
    if not response:
        return "I could not generate a safe response."

    cleaned = response.strip()

    replacements = {
        "this will definitely": "this may",
        "guaranteed": "likely",
        "always": "often",
        "never": "generally not",
    }

    lowered = cleaned.lower()
    for old, new in replacements.items():
        if old in lowered:
            cleaned = cleaned.replace(old, new)
            cleaned = cleaned.replace(old.capitalize(), new.capitalize())

    return cleaned


def apply_output_guardrails(response: str, original_message: str) -> str:
    safe_response = sanitize_output(response)

    if needs_safety_note(original_message):
        safe_response += (
            "\n\nNote: This is general fitness guidance and not medical advice. "
            "Please consult a qualified professional for injuries or health conditions."
        )

    return safe_response