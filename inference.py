import torch
from transformers import AutoTokenizer
from GPT_model import GPT2, GPT_CONFIG_124M  # Import your model

# Load tokenizer
model_path = "saved_models/gpt2_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load trained model
model = GPT2(GPT_CONFIG_124M)  # Initialize your custom model
model.load_state_dict(torch.load("saved_models/gpt2_finetuned.pth"))
model.eval()  # Set model to evaluation mode

# Function for text generation
def generate_text(prompt, max_tokens=50, temperature=0.7, top_k=10, top_p=0.95):
    #inputs = tokenizer(prompt, return_tensors="pt").input_ids

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,  # Force truncation
        max_length=GPT_CONFIG_124M["context_length"]  #  Match model's context length
    ).input_ids


    with torch.no_grad():
        output = model.generate(
            inputs,
            max_length=inputs.shape[1] + max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    user_prompt = input("Enter a prompt: ")
    generated_text = generate_text(user_prompt)
    print("\nGenerated Text:\n", generated_text)
