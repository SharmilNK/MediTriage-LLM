"""
custom_evaluation_example.py

Example script for:
- Loading the fine-tuned MediTriage LoRA adapter
- Running it on a custom evaluation set
- Computing a simple department accuracy metric

Usage (from repo root):
    python examples/custom_evaluation_example.py

Make sure you have:
- models/meditriage-mistral-lora/  (fine-tuned adapter + tokenizer)
- data/processed/custom_eval.json  (same schema as data/processed/test.json)
"""

import json
import os
from typing import List, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "models/meditriage-mistral-lora"
CUSTOM_PATH = "data/processed/custom_eval.json"
OUTPUT_PATH = "data/outputs/finetuned_custom_eval_predictions.json"


SYSTEM_PROMPT = (
    "You are a medical triage assistant. Analyze the following patient message "
    "and output a JSON object with exactly these keys: department, symptoms, "
    "condition, sentiment, urgency_level. "
    "Do not include any text outside the JSON object."
)

GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}


def load_finetuned_model():
    """Load base Mistral + LoRA adapter in 4-bit."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    return model, tokenizer


def build_prompt(patient_message: str) -> List[Dict[str, str]]:
    return [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT}\n\nPatient message: {patient_message}",
        }
    ]


def generate_prediction(model, tokenizer, patient_message: str) -> str:
    messages = build_prompt(patient_message)
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **GENERATION_CONFIG)
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_custom_eval():
    if not os.path.exists(CUSTOM_PATH):
        raise FileNotFoundError(
            f"Custom eval file not found at {CUSTOM_PATH}. "
            "Create it with the same schema as data/processed/test.json."
        )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Loading custom eval data from {CUSTOM_PATH}...")
    with open(CUSTOM_PATH, "r", encoding="utf-8") as f:
        eval_data: List[Dict[str, Any]] = json.load(f)
    print(f"Loaded {len(eval_data)} samples")

    print(f"Loading fine-tuned model from {ADAPTER_PATH}...")
    model, tokenizer = load_finetuned_model()
    print("Model loaded.")

    results = []
    for i, sample in enumerate(eval_data):
        patient_msg = sample["patient_message"]
        raw_output = generate_prediction(model, tokenizer, patient_msg)

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed = None

        results.append(
            {
                "id": i,
                "patient_message": patient_msg,
                "ground_truth": sample.get("ground_truth"),
                "raw_model_output": raw_output,
                "parsed_output": parsed,
                "valid_json": parsed is not None,
            }
        )

        if (i + 1) % 10 == 0:
            valid = sum(1 for r in results if r["valid_json"])
            print(f"[{i+1}/{len(eval_data)}] Valid JSON: {valid}/{len(results)}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved custom evaluation predictions to {OUTPUT_PATH}")

    # Simple example metric: department exact-match accuracy
    total = 0
    correct = 0
    for r in results:
        gt = r.get("ground_truth") or {}
        pred = r.get("parsed_output") or {}
        if "department" in gt and "department" in pred:
            total += 1
            if gt["department"] == pred["department"]:
                correct += 1

    if total > 0:
        acc = correct / total
        print(f"Department accuracy: {correct}/{total} = {acc:.3f}")
    else:
        print("No samples with department labels to evaluate.")


if __name__ == "__main__":
    run_custom_eval()

