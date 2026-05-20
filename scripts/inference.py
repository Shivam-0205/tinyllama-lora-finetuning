from transformers import pipeline
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained("./tinyllama-lora-model")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

prompt = "Machine learning is"

result = pipe(
    prompt,
    max_new_tokens=50
)

print(result[0]["generated_text"])
