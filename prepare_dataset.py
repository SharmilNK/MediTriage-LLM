"""
prepare_dataset.py

Loads finetuning_data.csv, maps fields to the target triage JSON schema,
formats each sample as a Mistral Instruct chat-template pair, and
performs a 75/15/15 stratified train/val/test split.

Input:
    data/raw/finetuning_data.csv  (columns: questions, output)

Output:
    data/processed/train.json
    data/processed/val.json
    data/processed/test.json
"""

import csv
import json
import os
import random
from typing import List, Dict, Tuple

SEED = 42
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
RAW_PATH = "data/raw/finetuning_data.csv"
TRAIN_PATH = "data/processed/train.json"
VAL_PATH = "data/processed/val.json"
TEST_PATH = "data/processed/test.json"

SYSTEM_PROMPT = (
    "You are a medical triage assistant. Analyze the following patient message "
    "and output a JSON object with exactly these keys: department, symptoms, "
    "condition, sentiment, urgency_level. "
    "Do not include any text outside the JSON object."
)


def load_raw_data(path: str) -> List[Dict]:
    """
    Load finetuning_data.csv into a list of dict rows.

    Expected columns:
        - questions: free-text patient message
        - output: JSON string with at least
          department, symptoms, condition, sentiment, urgency_level
    """
    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def map_to_target_schema(row: Dict) -> Dict:
    """
    Map a CSV row to the target triage JSON schema.

    The `output` column is expected to be a JSON object string like:

        {
          "department": "Pharmacology",
          "accepted_department": ["Pharmacology", "Internal Medicine", "Infectious Disease"],
          "symptoms": [],
          "condition": "Unknown",
          "sentiment": "Curious",
          "urgency_level": "Low"
        }

    Extra keys (e.g. accepted_department) are ignored for training.
    """
    raw_output = row["output"]
    parsed = json.loads(raw_output)

    # Symptoms might be an array already; if not, coerce to list.
    symptoms = parsed.get("symptoms", [])
    if not isinstance(symptoms, list):
        symptoms = [symptoms]

    return {
        "department": parsed["department"],
        "symptoms": symptoms,
        "condition": parsed.get("condition", "Unknown"),
        "sentiment": parsed.get("sentiment", "Unknown"),
        "urgency_level": parsed["urgency_level"],
    }


def format_instruction_pair(patient_message: str, target_json: Dict) -> Dict:
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


def stratified_split_3way(
    data: List[Dict], train_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Stratified 3-way split by urgency_level into train / val / test.
    """
    rng = random.Random(seed)

    # Group by urgency
    buckets: Dict[str, List[Dict]] = {}
    for item in data:
        key = item["ground_truth"]["urgency_level"]
        buckets.setdefault(key, []).append(item)

    train, val, test = [], [], []
    for urgency, items in sorted(buckets.items()):
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    # Shuffle the final lists so urgency groups aren't contiguous
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def main():
    raw_data = load_raw_data(RAW_PATH)
    print(f"Loaded {len(raw_data)} samples from {RAW_PATH}")

    # Convert each sample to instruction-tuning format
    formatted: List[Dict] = []
    for row in raw_data:
        target = map_to_target_schema(row)
        pair = format_instruction_pair(row["questions"], target)
        formatted.append(pair)

    print(f"Formatted {len(formatted)} instruction pairs")

    # Stratified 75/15/15 split
    train_data, val_data, test_data = stratified_split_3way(
        formatted, TRAIN_RATIO, VAL_RATIO, SEED
    )
    print(
        f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}"
    )

    # Print urgency distribution
    for split_name, split_data in [
        ("Train", train_data),
        ("Val", val_data),
        ("Test", test_data),
    ]:
        dist: Dict[str, int] = {}
        for item in split_data:
            u = item["ground_truth"]["urgency_level"]
            dist[u] = dist.get(u, 0) + 1
        print(f"  {split_name} urgency distribution: {dict(sorted(dist.items()))}")

    # Save
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(VAL_PATH, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    with open(TEST_PATH, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

    print(f"Saved train to {TRAIN_PATH}")
    print(f"Saved val to {VAL_PATH}")
    print(f"Saved test to {TEST_PATH}")


if __name__ == "__main__":
    main()
