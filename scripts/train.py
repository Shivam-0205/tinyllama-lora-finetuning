from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

from datasets import load_dataset, concatenate_datasets

import torch
import wandb

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

wandb.init(
    project="tinyllama-lora-project",
    name="tinyllama-training"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

tiny = load_dataset("roneneldan/TinyStories", split="train[:1000]")
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1000]")

combined_dataset = concatenate_datasets([tiny, wiki])

def clean(example):
    return {"text": example["text"]}

combined_dataset = combined_dataset.map(clean)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_dataset = combined_dataset.map(tokenize)

training_args = TrainingArguments(
    output_dir="./outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=100,
    save_total_limit=1,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="wandb"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("./tinyllama-lora-model")
tokenizer.save_pretrained("./tinyllama-lora-model")
