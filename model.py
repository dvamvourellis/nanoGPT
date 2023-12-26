import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    #implements the self-attention head
    def __init__(self, config):
        super().__init__()
        head_size = config["n_embd"] // config["n_head"]
        self.config = config
        self.key = nn.Linear(config["n_embd"], head_size, bias=False) #does matrix multiplication with fixed weights to map C-d input to head_size-d output
        self.query = nn.Linear(config["n_embd"], head_size, bias=False)
        self.value = nn.Linear(config["n_embd"], head_size, bias=False)
        #need to register the tril matrix cause its not model parameter
        self.register_buffer('tril', torch.tril(torch.ones(config["block_size"], config["block_size"])))

        #randomly zero out some attention weights
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        B,T,C = x.shape 
        k = self.key(x) # (B, T, 16) 
        q = self.query(x) # (B, T, 16)
        #make weight to be dependent on how similar is each token with the other ones in the sequence
        wei = q @ k.transpose(-2, -1) * C**-0.5# (B, T, 16) @ (B, 16, T) ---> (B, T, T) we scale by 1/sqrt(head_size) to make the softmax distribution less concentraded especially in the beginning

        #mask anything from the future - each token can attend to previous tokens only
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # set everything to -inf where tril = 0
        #normalize weights
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei)
        #calculate final weighted average - currently attending with equal to weight to previous tokens
        v = self.value(x) #v is head-dependent vlaue for each token learned by this one attention head
        out = wei @ v # (B, T, T) @ (B, T, head_size) ---> (B, T, head_size)
        return out 

class MultiHeadAttention(nn.Module):
    #implements the multi-headed attention
    def __init__(self, config):
        super().__init__()
        head_size = config["n_embd"] // config["n_head"]
        self.config = config
        self.heads = nn.ModuleList([Head(config) for _ in range(config["n_head"])])
        self.proj = nn.Linear(head_size * config["n_head"], config["n_embd"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x):
        #every head creates (head_size,) representation for each token that we want to concatenate
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    #implements a simple linear layer followed by non-linearity
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config["n_embd"], 4*config["n_embd"]),
            nn.ReLU(),
            nn.Linear(4*config["n_embd"], config["n_embd"]),
            nn.Dropout(config["dropout"])
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    #implements one transformer block
    def __init__(self, config):
        super().__init__()
        self.config = config
        #communication between tokens
        self.sa = MultiHeadAttention(config)
        #followed by computation 
        self.ffwd = FeedForward(config)
        #we normalize the features (i.e. per token) before every transformation to zero mean and unit variance
        self.ln1 = nn.LayerNorm(config["n_embd"])
        self.ln2 = nn.LayerNorm(config["n_embd"])
    
    def forward(self, x):
        #add residual connection to distribute gradient bakc to input - easier to optimize deep nets with residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# super simple bigram model
class GPTLanguageModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # each token directly reads off its embedding from lookup table
        self.token_embedding_table = nn.Embedding(config["vocab_size"], config["n_embd"])
        #create positional encoding for each positon up to block size
        self.position_embedding_table = nn.Embedding(config["block_size"], config["n_embd"])
        #create multiple transformer blocks each consisting by 4-headed attention layer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config["n_layer"])])
        self.ln_f = nn.LayerNorm(config["n_embd"]) # final layer norm
        self.lm_head = nn.Linear(config["n_embd"], config["vocab_size"])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        #input is the sum of token embeddings and positional embeddings
        x = tok_emb + pos_emb #broadcasting happens to get (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config["block_size"]:]
            # get the predictions
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature # becomes (B, C)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx