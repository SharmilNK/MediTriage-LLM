"""
finetune.py

Fine-tune Mistral 7B Instruct using QLoRA (4-bit) + PEFT/LoRA on the
medical triage dataset. Uses SFTTrainer from the trl library.

Input:
    data/processed/train.json  (from prepare_dataset.py)
    data/processed/val.json    (for validation)

Output:
    models/meditriage-mistral-lora/  (LoRA adapter weights)
    models/logs/training_log.json    (train/val losses for plotting)
    data/outputs/plots/train_val_loss.png
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
OUTPUT_DIR = "models/meditriage-mistral-lora"
LOGGING_DIR = "models/logs"
PLOTS_DIR = "data/outputs/plots"

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
    #test_dataset = load_dataset_from_path(TEST_PATH)
    print(
        f"Training samples: {len(train_dataset)} | "
        f"Val samples: {len(val_dataset)} | "
        #f"Test samples: {len(test_dataset)}"
    )

    # Load model + tokenizer + LoRA
    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = create_model_and_tokenizer()

    os.makedirs(LOGGING_DIR, exist_ok=True)

    # Training arguments
    # Note: the transformers version in Colab may not support newer
    # fields like `evaluation_strategy` / `load_best_model_at_end`,
    # so we keep this config minimal and call `trainer.evaluate`
    # manually after training.
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
        remove_unused_columns=False,
    )

    # SFTTrainer handles chat template formatting automatically.
    # Older trl versions don't support `max_seq_length` in the ctor,
    # so we only pass the arguments they accept.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("Starting fine-tuning...")
    train_result = trainer.train()

    # Evaluate on validation set for loss curves.
    # Important: do NOT pass eval_dataset here; SFTTrainer has already
    # prepared its own eval_dataset internally, and passing a raw
    # Dataset confuses the Trainer's column handling on older versions.
    print("Evaluating model on validation set...")
    val_metrics = trainer.evaluate(metric_key_prefix="eval")
    print(val_metrics)
    # Save training / evaluation metrics to a JSON log for plotting
    metrics_log = {
        "train_metrics": train_result.metrics,
        "val_metrics": val_metrics,
        "history": trainer.state.log_history,
    }
    metrics_path = os.path.join(LOGGING_DIR, "training_log.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)
    print(f"Saved training log to {metrics_path}")

    # Save loss plots for later inspection
    try:
        import matplotlib.pyplot as plt  # type: ignore

        os.makedirs(PLOTS_DIR, exist_ok=True)
        history = trainer.state.log_history

        steps = [h["step"] for h in history if "loss" in h]
        train_losses = [h["loss"] for h in history if "loss" in h]
        val_points = [(h["step"], h["eval_loss"]) for h in history if "eval_loss" in h]

        plt.figure(figsize=(6, 4))
        if steps and train_losses:
            plt.plot(steps, train_losses, label="train_loss")
        if val_points:
            val_steps, val_losses = zip(*val_points)
            plt.plot(val_steps, val_losses, "o-", label="val_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(PLOTS_DIR, "train_val_loss.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved loss plot to {plot_path}")
    except Exception as e:  # pragma: no cover
        print(f"Could not generate loss plot: {e}")

    # Save the LoRA adapter (not the full model)
    print(f"Saving LoRA adapter to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Fine-tuning complete!")
    print(f"Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
