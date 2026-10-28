import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
DATA_PATH = Path(os.environ.get("DATA_PATH", "data/train.jsonl"))
OUT_DIR = Path("output/smollm2-adapter")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "384"))


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATA_PATH}")

    rows = []
    for line in DATA_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        instruction = str(item["instruction"]).strip()
        response = str(item["response"]).strip()
        if instruction and response:
            rows.append({"instruction": instruction, "response": response})

    if len(rows) < 10:
        raise SystemExit("Dataset too small: provide at least 10 examples.")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    def tokenize(example):
        user_messages = [{"role": "user", "content": example["instruction"]}]
        full_messages = user_messages + [{"role": "assistant", "content": example["response"]}]

        prompt_text = tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tokenizer(
            prompt_text, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False
        )["input_ids"]
        full = tokenizer(
            full_text, truncation=True, max_length=MAX_LENGTH, add_special_tokens=False
        )
        input_ids = full["input_ids"]
        labels = list(input_ids)
        prompt_length = min(len(prompt_ids), len(labels))
        labels[:prompt_length] = [-100] * prompt_length
        if all(label == -100 for label in labels):
            labels[-1] = input_ids[-1]

        return {
            "input_ids": input_ids,
            "attention_mask": full["attention_mask"],
            "labels": labels,
        }

    dataset = Dataset.from_list(rows).map(
        tokenize,
        remove_columns=["instruction", "response"],
        desc="Tokenizing Indonesian dialogue data",
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    model.print_trainable_parameters()

    epochs = float(os.environ.get("TRAIN_EPOCHS", "3"))
    args = TrainingArguments(
        output_dir="output/smollm2-checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        use_cpu=True,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        optim="adamw_torch",
        remove_unused_columns=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding=True
        ),
    )
    trainer.train()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved SmolLM2 LoRA adapter to {OUT_DIR}")


if __name__ == "__main__":
    main()
