import torch 

def read_data(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text

def create_vocab(text):
    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, vocab_size

def encode(string, stoi):
    return [
        stoi[c] for c in string
    ]

def decode(tokens_list, itos):
    return "".join([itos[i] for i in tokens_list])

def prep_data_train_val(config):
    text = read_data(config['data_path'])
    stoi, itos, vocab_size = create_vocab(text)
    config['vocab_size'] = vocab_size
    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    n = int(config['train_size'] * len(data)) 
    train_data = data[:n]
    val_data = data[n:]

    data_dict = {}
    data_dict["stoi"] = stoi
    data_dict["itos"] = itos
    data_dict["vocab_size"] = vocab_size
    data_dict["train_data"] = train_data
    data_dict["val_data"] = val_data

    return data_dict
