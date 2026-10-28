import json
import os
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

BASE_MODEL = os.environ.get("BASE_MODEL", "google/gemma-3-270m-it")
DATA_PATH = Path("data/train.jsonl")
OUT_DIR = Path("output/adapter")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "384"))

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN secret is missing. Add it in repository Actions secrets.")
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}")

    rows = []
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            rows.append({"messages": [
                {"role": "user", "content": str(item["instruction"]).strip()},
                {"role": "assistant", "content": str(item["response"]).strip()},
            ]})
    if len(rows) < 10:
        raise SystemExit("Dataset too small: provide at least 10 examples.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, token=token, torch_dtype="auto", low_cpu_mem_usage=True
    )
    model.config.use_cache = False

    def tokenize(example):
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        encoded = tokenizer(text, truncation=True, max_length=MAX_LENGTH, padding=False)
        encoded["labels"] = list(encoded["input_ids"])
        return encoded

    dataset = Dataset.from_list(rows).map(tokenize, remove_columns=["messages"], desc="Tokenizing dataset")
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear",
    ))
    model.print_trainable_parameters()

    epochs = float(os.environ.get("TRAIN_EPOCHS", "3"))
    args = TrainingArguments(
        output_dir="output/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        use_cpu=True,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved LoRA adapter to {OUT_DIR}")

if __name__ == "__main__":
    main()
