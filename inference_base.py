"""
inference_base.py

Run the BASE (pre-fine-tuning) Mistral 7B Instruct model on the test set
to establish baseline performance. Saves predictions so the evaluation
team can compare against the fine-tuned model.

Output:
    data/outputs/base_model_predictions.json
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
TEST_PATH = "data/processed/test.json"
OUTPUT_PATH = "data/outputs/base_model_predictions.json"

SYSTEM_PROMPT = (
    "You are a medical triage assistant. Analyze the following patient message "
    "and output a JSON object with exactly these keys: department, symptoms, "
    "condition, sentiment, urgency_level. "
    "Do not include any text outside the JSON object."
)

# Generation config — low temperature for deterministic JSON
GENERATION_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.1,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}


def load_model_and_tokenizer():
    """Load base Mistral 7B with 4-bit quantization."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    return model, tokenizer


def build_prompt(patient_message: str) -> list[dict]:
    """Build chat messages in the same format used for training."""
    return [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT}\n\nPatient message: {patient_message}",
        }
    ]


def generate_prediction(model, tokenizer, patient_message: str) -> str:
    """Generate a single prediction from the model."""
    messages = build_prompt(patient_message)
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, **GENERATION_CONFIG)

    # Decode only the new tokens (skip the input)
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    # Load test data
    with open(TEST_PATH, "r") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test samples")

    # Load model
    print(f"Loading base model: {MODEL_ID}")
    model, tokenizer = load_model_and_tokenizer()
    print("Model loaded successfully")

    # Run inference
    results = []
    for i, sample in enumerate(test_data):
        patient_msg = sample["patient_message"]
        raw_output = generate_prediction(model, tokenizer, patient_msg)

        # Try to parse as JSON
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed = None

        result = {
            "id": i,
            "patient_message": patient_msg,
            "ground_truth": sample["ground_truth"],
            "raw_model_output": raw_output,
            "parsed_output": parsed,
            "valid_json": parsed is not None,
        }
        results.append(result)

        if (i + 1) % 10 == 0:
            valid = sum(1 for r in results if r["valid_json"])
            print(f"  [{i+1}/{len(test_data)}] Valid JSON: {valid}/{len(results)}")

    # Summary
    valid_count = sum(1 for r in results if r["valid_json"])
    print(f"\nDone. Valid JSON: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
