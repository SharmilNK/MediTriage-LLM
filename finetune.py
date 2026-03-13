"""
finetune.py

Fine-tune Mistral 7B Instruct using QLoRA (4-bit) + PEFT/LoRA on the
medical triage dataset. Uses SFTTrainer from the trl library.

Input:
    data/processed/train.json  (from prepare_dataset.py)
    data/processed/val.json    (for validation / early stopping)

Output:
    models/meditriage-mistral-lora/  (LoRA adapter weights)
    models/logs/training_log.json    (train/val/test losses for plotting)
"""

import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
TRAIN_PATH = "data/processed/train.json"
VAL_PATH = "data/processed/val.json"
TEST_PATH = "data/processed/test.json"
OUTPUT_DIR = "models/meditriage-mistral-lora"
LOGGING_DIR = "models/logs"

# ── LoRA Config ────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ── Training Hyperparameters ───────────────────────────────────────────
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4        # effective batch size = 4 * 4 = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
MAX_SEQ_LENGTH = 1024
SEED = 42


def load_dataset_from_path(path: str) -> Dataset:
    """Load data from JSON file and convert to HuggingFace Dataset."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # SFTTrainer expects a "messages" column in chat format
    records = [{"messages": item["messages"]} for item in data]
    return Dataset.from_list(records)


def create_model_and_tokenizer():
    """Load Mistral 7B with 4-bit quantization and attach LoRA."""
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Prepare for k-bit training (freezes base, enables gradient checkpointing)
    model = prepare_model_for_kbit_training(model)

    # Attach LoRA adapters
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def main():
    # Load data
    train_dataset = load_dataset_from_path(TRAIN_PATH)
    val_dataset = load_dataset_from_path(VAL_PATH)
    test_dataset = load_dataset_from_path(TEST_PATH)
    print(
        f"Training samples: {len(train_dataset)} | "
        f"Val samples: {len(val_dataset)} | "
        f"Test samples: {len(test_dataset)}"
    )

    # Load model + tokenizer + LoRA
    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = create_model_and_tokenizer()

    os.makedirs(LOGGING_DIR, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_dir=LOGGING_DIR,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=SEED,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # SFTTrainer handles chat template formatting automatically
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Train
    print("Starting fine-tuning...")
    train_result = trainer.train()

    # Evaluate on validation and test sets for loss curves
    print("Evaluating best model on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="eval")
    print(val_metrics)

    print("Evaluating best model on test set...")
    test_metrics = trainer.evaluate(
        eval_dataset=test_dataset,
        metric_key_prefix="test",
    )
    print(test_metrics)

    # Save training / evaluation metrics to a JSON log for plotting
    metrics_log = {
        "train_metrics": train_result.metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "history": trainer.state.log_history,
    }
    metrics_path = os.path.join(LOGGING_DIR, "training_log.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Saved training log to {metrics_path}")

    # Save the LoRA adapter (not the full model)
    print(f"Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Fine-tuning complete!")
    print(f"Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
