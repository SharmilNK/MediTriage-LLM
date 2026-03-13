"""
prepare_dataset.py

Loads raw_dataset.json, maps fields to the target triage JSON schema,
formats each sample as a Mistral Instruct chat-template pair, and
performs an 80/20 stratified train/test split.

Output:
    data/processed/train.json
    data/processed/test.json
"""

import json
import os
import random

SEED = 42
TRAIN_RATIO = 0.8
RAW_PATH = "data/raw/raw_dataset.json"
TRAIN_PATH = "data/processed/train.json"
TEST_PATH = "data/processed/test.json"

SYSTEM_PROMPT = (
    "You are a medical triage assistant. Analyze the following patient message "
    "and output a JSON object with exactly these keys: department, symptoms, "
    "condition, sentiment, urgency_level. "
    "Do not include any text outside the JSON object."
)


def load_raw_data(path: str) -> list[dict]:
    with open(path, "r") as f:
        return json.load(f)


def map_to_target_schema(sample: dict) -> dict:
    """Extract and remap fields to the target output schema."""
    return {
        "department": sample["labels"]["department"],
        "symptoms": sample["metadata"]["symptoms_used"],
        "condition": sample["metadata"]["likely_condition"],
        "sentiment": sample["labels"]["sentiment"],
        "urgency_level": sample["labels"]["urgency"],
    }


def format_instruction_pair(patient_message: str, target_json: dict) -> dict:
    """Format as a Mistral Instruct chat-template training example."""
    user_content = f"{SYSTEM_PROMPT}\n\nPatient message: {patient_message}"
    assistant_content = json.dumps(target_json, indent=2)

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        # Keep ground truth separately for easy evaluation later
        "ground_truth": target_json,
        "patient_message": patient_message,
    }


def stratified_split(data: list[dict], ratio: float, seed: int):
    """Split data stratified by urgency_level."""
    rng = random.Random(seed)

    # Group by urgency
    buckets: dict[str, list[dict]] = {}
    for item in data:
        key = item["ground_truth"]["urgency_level"]
        buckets.setdefault(key, []).append(item)

    train, test = [], []
    for urgency, items in sorted(buckets.items()):
        rng.shuffle(items)
        split_idx = int(len(items) * ratio)
        train.extend(items[:split_idx])
        test.extend(items[split_idx:])

    # Shuffle the final lists so urgency groups aren't contiguous
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def main():
    raw_data = load_raw_data(RAW_PATH)
    print(f"Loaded {len(raw_data)} samples from {RAW_PATH}")

    # Convert each sample to instruction-tuning format
    formatted = []
    for sample in raw_data:
        target = map_to_target_schema(sample)
        pair = format_instruction_pair(sample["patient_message"], target)
        formatted.append(pair)

    print(f"Formatted {len(formatted)} instruction pairs")

    # Stratified 80/20 split
    train_data, test_data = stratified_split(formatted, TRAIN_RATIO, SEED)
    print(f"Train: {len(train_data)} | Test: {len(test_data)}")

    # Print urgency distribution
    for split_name, split_data in [("Train", train_data), ("Test", test_data)]:
        dist: dict[str, int] = {}
        for item in split_data:
            u = item["ground_truth"]["urgency_level"]
            dist[u] = dist.get(u, 0) + 1
        print(f"  {split_name} urgency distribution: {dict(sorted(dist.items()))}")

    # Save
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    with open(TRAIN_PATH, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(TEST_PATH, "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Saved train to {TRAIN_PATH}")
    print(f"Saved test to {TEST_PATH}")


if __name__ == "__main__":
    main()
