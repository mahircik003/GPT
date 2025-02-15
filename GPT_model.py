from transformers import GPT2Tokenizer

# tokenizer is here
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

import torch
import torch.nn as nn
import math

TEMP = 1.5


GPT_CONFIG = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 64, # Context length     1024
    "emb_dim": 512,         # Embedding dimension         768
    "n_heads": 4,          # Number of attention heads    12
    "n_layers": 4,         # Number of layers             12
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}



# Embedding Layer
class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        # Initialize the embedding layer with the specified vocabulary size and embedding dimension
        self.embed = nn.Embedding(vocab_size, embed_size)
    
    def forward(self, x):
        # Forward pass: convert input token IDs to their corresponding embeddings
        return self.embed(x)


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_seq_length=32):
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
        
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_seq_length, embed_size)

    def forward(self, x):
        # Add the positional encodings to the input embeddings
        return x + self.pe[:, :x.size(1)]

# Multi-Head Attention
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


# Feed-Forward Network
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


# Transformer Block
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


# GPT Model
import torch.nn.functional as F

class GPT_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = Embedding(config["vocab_size"], config["emb_dim"])
        self.positional_encoding = PositionalEncoding(config["emb_dim"], config["context_length"])
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(config["emb_dim"], config["n_heads"], config["emb_dim"] * 4, config["drop_rate"], config["qkv_bias"])
            for _ in range(config["n_layers"])
        ])
        self.fc_out = nn.Linear(config["emb_dim"], config["vocab_size"])
        self.dropout = nn.Dropout(config["drop_rate"])

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.dropout(self.positional_encoding(self.embedding(input_ids)))

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0

        for block in self.transformer_blocks:
            x = block(x, attention_mask)

        logits = self.fc_out(x)

        loss = None
        if labels is not None:
            # Use a default pad token ID of -100 if none is specified
            pad_token_id = -100 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=pad_token_id)

        return {"loss": loss, "logits": logits} if loss is not None else logits


model = GPT_model(GPT_CONFIG)


input_ids = torch.randint(0, GPT_CONFIG["vocab_size"], (2, 64))

output = model(input_ids)


assert output.shape == (2, 64, GPT_CONFIG["vocab_size"]), "GPT-2 Model shape mismatch"

def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():  # Disable gradient calculation for inference
            logits = model(idx_cond) 



        
        logits = logits[:, -1, :] 
        temperature = TEMP
        logits = logits / temperature    # scaling by temperature

        # Convert logits to probabilities using softmax
        probas = torch.softmax(logits, dim=-1) 


        idx_next = torch.multinomial(probas, num_samples=1)  # generates a sample a picks from it (instead of deterministic result)


        idx = torch.cat((idx, idx_next), dim=1) 

    return idx 
