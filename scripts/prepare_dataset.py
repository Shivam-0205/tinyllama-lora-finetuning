from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tiny = load_dataset(
    "roneneldan/TinyStories",
    split="train[:1000]"
)

wiki = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="train[:1000]"
)

agnews = load_dataset(
    "ag_news",
    split="train[:500]"
)

combined_dataset = concatenate_datasets([
    tiny,
    wiki,
    agnews
])

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

print(tokenized_dataset)
