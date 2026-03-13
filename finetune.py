"""
finetune.py

Fine-tune Mistral 7B Instruct using QLoRA (4-bit) + PEFT/LoRA on the
medical triage dataset. Uses SFTTrainer from the trl library.

Input:
    data/processed/train.json  (from prepare_dataset.py)

Output:
    models/meditriage-mistral-lora/  (LoRA adapter weights)
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


def load_train_data(path: str) -> Dataset:
    """Load training data and convert to HuggingFace Dataset."""
    with open(path, "r") as f:
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
    train_dataset = load_train_data(TRAIN_PATH)
    print(f"Training samples: {len(train_dataset)}")

    # Load model + tokenizer + LoRA
    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = create_model_and_tokenizer()

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
        save_strategy="epoch",
        save_total_limit=2,
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
        processing_class=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    # Train
    print("Starting fine-tuning...")
    trainer.train()

    # Save the LoRA adapter (not the full model)
    print(f"Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Fine-tuning complete!")
    print(f"Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
