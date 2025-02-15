import os
from itertools import chain
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
)

device = "cuda" if torch.cuda.is_available() else "cpu"
from GPT_model import GPT_model, GPT_CONFIG

model = GPT_model(GPT_CONFIG).to(device)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

TOKENIZED_DATASET_PATH = "bookcorpus/chunked_ds"

def tokenize_function(example):
    encodings = tokenizer(
        example["text"],
        truncation=True,
        max_length=GPT_CONFIG["context_length"],
        padding="max_length",
        return_tensors=None
    )
    encodings["labels"] = encodings["input_ids"][:]
    return encodings

def group_texts(examples):
    concatenated = {
        k: sum(examples[k], []) 
        for k in examples.keys()
    }
    
    total_length = len(concatenated["input_ids"])
    block_size = GPT_CONFIG["context_length"]
    total_length = (total_length // block_size) * block_size
    
    result = {
        k: [t[i:i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    
    return result

if not os.path.exists(TOKENIZED_DATASET_PATH):
    print("Tokenizing dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    # Create train/test split
    dataset = dataset["train"].train_test_split(test_size=0.1)
    
    tokenized_ds = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)
    chunked_ds = tokenized_ds.map(group_texts, batched=True, batch_size=1000)
    
    print("Saving tokenized dataset...")
    chunked_ds.save_to_disk(TOKENIZED_DATASET_PATH)
else:
    print("Loading pre-tokenized dataset...")
    chunked_ds = load_from_disk(TOKENIZED_DATASET_PATH)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir='gpt2_finetuned/',
    evaluation_strategy="steps",
    eval_steps=500,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=500,
    save_steps=5000,
    save_total_limit=5,
    max_grad_norm=1.0,
    report_to='none',
    remove_unused_columns=False
)

print(chunked_ds["train"].column_names)
print(chunked_ds["train"][0])

trainer = Trainer(
    model=model.to(device),
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=chunked_ds["train"],
    eval_dataset=chunked_ds["test"],
    data_collator=data_collator
)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
max_steps = 40000

validate_every = 500 
best_val_loss = float('inf')

for step, batch in enumerate(chunked_ds["train"]):
    if step >= max_steps:
        break
    optimizer.zero_grad()
    
    input_ids = torch.tensor(batch['input_ids']).to(device)
    labels = torch.tensor(batch['labels']).to(device)
    
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    print(f"Step {step}, Loss: {loss.item()}")
    
    if torch.isnan(loss):
        print("NaN loss detected!")
        break
        
    loss.backward()
    
    # Print gradient norms
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            if torch.isnan(grad_norm):
                print(f"NaN gradient in {name}")
            
    optimizer.step()

    print(f"Step {step}, Train Loss: {loss.item()}")
    
    if step % validate_every == 0 and step > 0:
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for val_batch in chunked_ds["test"]:
                val_input_ids = torch.tensor(val_batch['input_ids']).to(device)
                val_labels = torch.tensor(val_batch['labels']).to(device)
                val_outputs = model(val_input_ids, labels=val_labels)
                val_losses.append(val_outputs['loss'].item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Step {step}, Validation Loss: {avg_val_loss}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "saved_models/gpt2_best.pth")
        
        model.train()


os.makedirs("saved_models", exist_ok=True)
torch.save(model.state_dict(), "saved_models/gpt2_finetuned.pth")
tokenizer.save_pretrained("saved_models/gpt2_finetuned")

print("Training complete. Model saved in 'saved_models/gpt2_finetuned'.")
