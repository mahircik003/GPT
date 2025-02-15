import torch
from transformers import AutoTokenizer
from GPT_model import GPT_model, GPT_CONFIG, generate_text_simple

import os


# Load tokenizer
if os.path.exists("saved_models/gpt2_finetuned"):
    tokenizer = AutoTokenizer.from_pretrained("saved_models/gpt2_finetuned")
else:
    print("Fine-tuned tokenizer not found. Using default GPT-2 tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT_model(GPT_CONFIG).to(device)

if os.path.exists("saved_models/gpt2_finetuned.pth"):
    model.load_state_dict(torch.load("saved_models/gpt2_finetuned.pth", map_location=device))
    print("Loaded fine-tuned model.")
else:
    print("Warning: No trained model found. Running with randomly initialized weights.")


model.eval()

# Function for text generation
def generate_text(prompt, max_tokens=50, temperature=0.7, top_k=10, top_p=0.95):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=GPT_CONFIG["context_length"]
    ).input_ids.to(device) 

    with torch.no_grad():
        output = generate_text_simple(  
            model, inputs, max_tokens, GPT_CONFIG["context_length"]
        )

    return tokenizer.decode(output.squeeze(0).tolist(), skip_special_tokens=True)  


if __name__ == "__main__":
    user_prompt = input("Enter a prompt: ")
    generated_text = generate_text(user_prompt)
    print("\nGenerated Text:\n", generated_text)
