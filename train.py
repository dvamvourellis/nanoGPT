import torch 
from data_prep_utils import prep_data_train_val, encode, decode
from model import GPTLanguageModel
import yaml

torch.manual_seed(1337)

#read yaml file
with open('config.yaml') as file:
  config = yaml.safe_load(file)

#find the device to load data and model
device = "cuda" if torch.cuda.is_available() else "cpu"

#unpack hyperparameters
block_size = config["block_size"]
batch_size = config["batch_size"]
eval_iters = config["eval_iters"]
learning_rate = config["learning_rate"]
max_iters = config["max_iters"]
eval_interval = config["eval_interval"]

#load data
data_dict = prep_data_train_val(config)
stoi = data_dict["stoi"]
itos = data_dict["itos"]
vocab_size = data_dict["vocab_size"]
train_data = data_dict["train_data"]
val_data = data_dict["val_data"] 

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()  # we are not calling backward in this function - memory optimization
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPTLanguageModel(config)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


m.eval()
# do unconditional generation starting from \n character
idx = torch.zeros((1, 1), dtype=torch.long)
# generate a document of 100 tokens
doc_generated = m.generate(idx, max_new_tokens=100)
doc_decoded = decode(doc_generated.tolist()[0], itos)
print(doc_decoded)
