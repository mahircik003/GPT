from transformers import GPT2Tokenizer

# tokenizer is here
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

import torch
import torch.nn as nn
import math


GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 64, # Context length     1024
    "emb_dim": 512,         # Embedding dimension         768
    "n_heads": 4,          # Number of attention heads    12
    "n_layers": 4,         # Number of layers             12
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}



# 1. Embedding Layer
class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        # Initialize the embedding layer with the specified vocabulary size and embedding dimension
        self.embed = nn.Embedding(vocab_size, embed_size)
    
    def forward(self, x):
        # Forward pass: convert input token IDs to their corresponding embeddings
        return self.embed(x)


# 2. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_seq_length=512):
        super().__init__()
        # Initialize a tensor to hold the positional encodings
        pe = torch.zeros(max_seq_length, embed_size)
        
        # Create a tensor for positions (0 to max_seq_length)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term for the sine and cosine functions
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)  # Sine for even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Cosine for odd indices
        
        # Register the positional encodings as a buffer (not a model parameter)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_seq_length, embed_size)

    def forward(self, x):
        # Add the positional encodings to the input embeddings
        return x + self.pe[:, :x.size(1)]

# A. Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, qkv_bias=False):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        self.query = nn.Linear(embed_size, embed_size, bias=qkv_bias)
        self.key = nn.Linear(embed_size, embed_size, bias=qkv_bias)
        self.value = nn.Linear(embed_size, embed_size, bias=qkv_bias)
        self.out = nn.Linear(embed_size, embed_size)
        
    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(attention, dim=-1)
        
        out = torch.matmul(attention, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        return self.out(out)


# 5. Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size):
        super().__init__()
        # First linear layer that transforms input from embedding size to hidden size
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        # Second linear layer that transforms from hidden size back to embedding size
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        # GELU activation function
        self.gelu = nn.GELU()

    def forward(self, x):
        # Forward pass: apply the first linear layer, then GELU activation, and finally the second linear layer
        return self.fc2(self.gelu(self.fc1(x)))


# 6. Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.1, qkv_bias=False):
        super().__init__()
        # Initialize the multi-head attention layer
        self.mha = MultiHeadAttention(embed_size, num_heads, qkv_bias)
        # Initialize the feed-forward network
        self.ff = FeedForward(embed_size, ff_hidden_size)
        # Initialize layer normalization for the attention output
        self.ln1 = nn.LayerNorm(embed_size)
        # Initialize layer normalization for the feed-forward output
        self.ln2 = nn.LayerNorm(embed_size)
        # Initialize dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Apply multi-head attention and add the residual connection, followed by layer normalization
        attention_output = self.ln1(x + self.dropout(self.mha(x, mask)))
        # Apply feed-forward network and add the residual connection, followed by layer normalization
        ff_output = self.ln2(attention_output + self.dropout(self.ff(attention_output)))
        return ff_output


# 7. GPT-2 Model
class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Initialize the embedding layer to convert token IDs to embeddings
        self.embedding = Embedding(config["vocab_size"], config["emb_dim"])
        
        # Initialize positional encoding to add positional information to embeddings
        self.positional_encoding = PositionalEncoding(config["emb_dim"], config["context_length"])
        
        # Create a list of transformer blocks
        self.transformer_blocks = nn.ModuleList([
            # Each transformer block consists of multi-head attention and feed-forward layers
            TransformerBlock(config["emb_dim"], config["n_heads"], config["emb_dim"] * 4, config["drop_rate"], config["qkv_bias"])
            for _ in range(config["n_layers"])  # Repeat for the number of layers specified in the config
        ])
        
        # Final linear layer to project the output back to the vocabulary size for logits
        self.fc_out = nn.Linear(config["emb_dim"], config["vocab_size"])
        
        # Dropout layer for regularization
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, x, mask=None):
        # Step 1: Convert input token IDs to embeddings and add positional encodings
        x = self.dropout(self.positional_encoding(self.embedding(x)))
        
        # Step 2: Pass the embeddings through each transformer block
        for block in self.transformer_blocks:
            x = block(x, mask)  # Apply the transformer block with optional masking
        
        # Step 3: Project the final output to the vocabulary size
        return self.fc_out(x)  # Shape: (batch_size, seq_length, vocab_size)

# Test GPT-2 Model
# Create an instance of the GPT-2 model using the configuration values
model = GPT2(GPT_CONFIG_124M)

# Generate random input token IDs with shape (batch_size, seq_length)
input_ids = torch.randint(0, GPT_CONFIG_124M["vocab_size"], (2, 64))

# Apply the model to the input token IDs
output = model(input_ids)

# Print the shape of the output from the model
#print(f"GPT-2 Model output shape: {output}")

# Assert that the output shape matches the expected shape
assert output.shape == (2, 64, GPT_CONFIG_124M["vocab_size"]), "GPT-2 Model shape mismatch"

def generate_text_simple(model, idx, max_new_tokens, context_size):

    # eos_token_id = tokenizer.eos_token_id  # GPT-2 uses <|endoftext|> as EOS
    # if eos_token_id is None:
    #     eos_token_id = 50256  # Manually set GPT-2's default EOS token


    # Loop to generate the specified number of new tokens

    for _ in range(max_new_tokens):
    #while(True):
        # Prepare the context
        # Crop the current context to the last 'context_size' tokens

        idx_cond = idx[:, -context_size:]  # Shape: (batch, context_size)

        # Get model predictions
        with torch.no_grad():  # Disable gradient calculation for inference
            logits = model(idx_cond)  # Shape: (batch, n_tokens, vocab_size)

        # Focus on the last time step's predictions
        logits = logits[:, -1, :]  # Shape: (batch, vocab_size)

        # Convert logits to probabilities using softmax
        probas = torch.softmax(logits, dim=-1)  # Shape: (batch, vocab_size)

        # Get the index of the token with the highest probability
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # Shape: (batch, 1)

        # **Stopping Condition: If EOS token is generated, stop immediately**
        # if idx_next.item() == eos_token_id:
        #     break

        # Append the predicted token index to the sequence
        idx = torch.cat((idx, idx_next), dim=1)  # Shape: (batch, n_tokens + 1)

    return idx  # Return the updated sequence of token indices

# Initial context for text generation
start_context = "I want to "

# Step 1: Encode the initial context into token indices
encoded = tokenizer.encode(start_context)
print("Encoded:", encoded)

# Convert the encoded list into a tensor and add a batch dimension
encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Shape: (1, n_tokens)

# Set the model to evaluation mode to disable dropout -- dropout has to be disabled during performance
model.eval()

# Step 3: Generate new tokens based on the initial context
out = generate_text_simple(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=50, 
    context_size=GPT_CONFIG_124M["context_length"]
)

# Step 4: Print the output tensor and its length
print("Output:", out)
print("Output length:", len(out[0]))

# Step 5: Decode the generated token indices back into text
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print("Decoded text:", decoded_text)