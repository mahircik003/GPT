import os
from itertools import chain
import torch
from datasets import load_dataset, load_from_disk
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer)

# Import your custom model
from GPT_model import GPT2, GPT_CONFIG_124M

# Load dataset (BookCorpus)
#dataset = load_dataset("bookcorpus", trust_remote_code=True)

dataset = load_dataset("roneneldan/TinyStories")


# Split dataset
dataset = dataset['train'].train_test_split(test_size=0.0015)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Tokenize function
# def tokenize_function(example):
#     return tokenizer(example["text"])

def tokenize_function(example):
    return tokenizer(
        example["text"],
        truncation=True,  #  Ensure sequences don't exceed max length
        max_length=GPT_CONFIG_124M["context_length"],  #  Force truncation
        padding="max_length"  # Optional: Pads shorter sequences
    )




tokenized_ds = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Save tokenized dataset (optional)
tokenized_ds.save_to_disk('bookcorpus/tokenized_ds')

# Concatenate tokenized examples into long sequences
def concat(examples):
    examples["input_ids"] = [list(chain.from_iterable(examples["input_ids"]))]
    examples["attention_mask"] = [list(chain.from_iterable(examples["attention_mask"]))]
    return examples

concated_ds = tokenized_ds.map(concat, batched=True, batch_size=1000000, num_proc=8)

# Chunk sequences into context-size pieces
def chunk(examples):
    chunk_size = GPT_CONFIG_124M["context_length"]
    input_ids = examples["input_ids"][0]
    attention_mask = examples["attention_mask"][0]
    input_ids_truncated, attention_mask_truncated = [], []

    for i in range(0, len(input_ids), chunk_size):
        chunk = input_ids[i:i+chunk_size]
        if len(chunk) == chunk_size:
            input_ids_truncated.append(chunk)
            attention_mask_truncated.append(attention_mask[i:i+chunk_size])

    examples["input_ids"] = input_ids_truncated
    examples["attention_mask"] = attention_mask_truncated
    return examples

chunked_ds = concated_ds.map(chunk, batched=True, batch_size=2, num_proc=2)

# Save preprocessed dataset
chunked_ds.save_to_disk('bookcorpus/chunked_ds')

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Load **your custom GPT model**
model = GPT2(GPT_CONFIG_124M)

# Define training arguments
training_args = TrainingArguments(
    output_dir='gpt2_finetuned/',
    evaluation_strategy="steps",
    eval_steps=500,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=2.5e-4,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=500,
    save_steps=5000,
    save_total_limit=5,
    report_to='none',  # Disable WandB unless needed
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=chunked_ds["train"],
    eval_dataset=chunked_ds["test"],
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save final trained model
os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/gpt2_finetuned.pth")
tokenizer.save_pretrained("saved_models/gpt2_finetuned")

print("Training complete. Model saved in 'saved_models/gpt2_finetuned'.")
