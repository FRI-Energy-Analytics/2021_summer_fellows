"""
Training model and forward passes
"""

import math
import time
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torchtext.data.utils import get_tokenizer  # type:ignore
from torchtext.vocab import Vocab  # type:ignore


class PositionalEncoding(nn.Module):
    """
    The PositionalEncoding injects some information about the relative and
    absoute positon of the tokens in sequence
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]  # type:ignore

        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer model with encoder and decoder (might change to encoder only later on).
    """

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        """
        Generate attention mask using triu (triangle) attention
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        """
        Normalize mean and standarad deviation of weights
        """
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        """
        Full forward pass
        """
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


device = torch.device("cpu")

bptt = 10
shift = 1


def get_batch(source, i):
    """
    Gets a batch shifted over by shift length
    """
    seq_len = min(bptt, len(source) - shift - i)
    data = source[i : i + seq_len]
    target = source[i + shift : i + shift + seq_len].reshape(-1)
    return data, target


def batchify(data, bsz):
    """
    Puts the data evenly into batches of bsz size
    """
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batxches.
    data = data.view(bsz, -1)
    return data.to(device)


def data_process(raw_text_iter):
    """
    Converts the data into a latent dimension using the given vocab
    """
    data = [
        torch.tensor(
            [vocab.stoi[token] for token in tokenizer(str(item))], dtype=torch.long
        )
        for item in raw_text_iter
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def conv(raw_output):
    """
    Converts the latent dimension into data again the opposite of data_process
    """
    return torch.tensor([int(vocab.itos[x]) for x in raw_output.view(-1)])


def convert_outpupt_to_tensor(output):
    """
    Converts the latent dimension into data again the opposite of data_process
    This version also accounts for the probability outputs using torch.argmax()
    """
    imm = [
        [vocab.itos[torch.argmax(prob).item()] for prob in result] for result in output
    ]
    ret = []
    for row in imm:
        new_row = []
        for i in row:
            if i == "<pad>":
                new_row.append(0)
            elif i == "<unk>":
                new_row.append(-1)
            else:
                new_row.append(int(i))
        ret.append(new_row)

    return torch.tensor(ret)


def train():
    """
    Trains the model for `epoch` iterations
    """
    model.train()  # Turn on the train mode
    total_loss = 0.0
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    log_interval = 200
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # Fetch Data
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()

        # Genearte attention mask
        if data.size(0) != bptt:
            src_mask = model.generate_square_subsequent_mask(data.size(0)).to(device)

        output = model(data, src_mask)

        # Backpropogate
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()

        # Logging
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(
                "| epoch {:3d} | {:5d}/{:5d} batches | "
                "lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // bptt,
                    scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, data_source):
    """
    Evaluates the training loss of the given model
    """
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.0
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = model.generate_square_subsequent_mask(data.size(0)).to(
                    device
                )
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


if __name__ == "__main__":

    wells = pd.read_csv("export_csv/2018.csv", index_col=[1, 0])

    # Lets choose a particular well to start off with
    # Our model only trains on single well, we NEED to expand
    # this futher for multiple time series
    well = wells.loc[314]  # type:ignore
    well = well[~well["gamma"].isna()]

    # For now we will use a tokenizer for basic_english on the integer inputs
    # We will change this for later when we create our own latent space
    x = well["gamma"]
    counter = Counter()
    tokenizer = get_tokenizer("basic_english")
    for line in x:
        line = str(int(line))
        counter.update(tokenizer(line))
    vocab = Vocab(counter)

    # Lets break it down using test train val split
    x_train, x_val, x_test = x[:1000:5], x[1000:1250:5], x[1250:3899:5]
    # Process data
    train_data = data_process(x_train.to_numpy(dtype="int64"))
    val_data = data_process(x_val.to_numpy(dtype="int64"))
    test_data = data_process(x_test.to_numpy(dtype="int64"))

    batch_size = 10
    eval_batch_size = 10
    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, eval_batch_size)
    test_data = batchify(test_data, eval_batch_size)

    best_val_loss = float("inf")
    epochs = 20  # The number of epochs
    best_model = None

    ntokens = len(vocab.stoi)  # the size of vocabulary

    # Define our TransformerModel
    emsize = 200  # embedding dimension
    nhid = (
        200  # the dimension of the feedforward network model in nn.TransformerEncoder
    )
    nlayers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    nhead = 4  # the number of heads in the multiheadattention models
    dropout = 0.2  # the dropout value
    model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 1.0, gamma=0.95  # type:ignore
    )

    # Training loop
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        print("-" * 89)
        print(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
            "valid ppl {:8.2f}".format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
            )
        )
        print("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

    # Combine output predicted data into full_output
    data_source = test_data
    full_output = []
    src_mask = torch.ones(bptt, bptt)
    with torch.no_grad():
        for i in range(0, data_source.size(0), bptt):
            data, targets = get_batch(data_source, i)
            if data.size(0) != bptt:
                src_mask = best_model.generate_square_subsequent_mask(data.size(0)).to(
                    device
                )
            output = model(data, src_mask)
            full_output.append(output)
    full_output = torch.cat(full_output)
    full_output = convert_outpupt_to_tensor(full_output)

    # Export results into dataframe (slices are used due to size differences)
    results = pd.DataFrame(
        {
            "Output": [t.item() for t in full_output.view(-1)],
            "Input": conv(test_data.view(-1))[105:],
        },
        index=pd.Index(name="depth", data=well[1250:3899:5].index.values[107:]),
    )

    # Lets plot the results to see the final output
    fig = plt.figure(figsize=(8, 10))
    x = well["gamma"].values

    ax = plt.gca()
    ax.plot(results["Output"], results.index.values, label="Prediction", color="red")
    ax.plot(
        well[:1000:5]["gamma"], well[:1000:5].index.values, label="Train", color="gray"
    )
    ax.plot(
        well[1000:1250:5]["gamma"],
        well[1000:1250:5].index.values,
        label="Validation",
        color="yellow",
    )
    ax.plot(
        well[1250::5]["gamma"], well[1250::5].index.values, label="Target", color="blue"
    )
    ax.legend()
    ax.invert_yaxis()
    plt.show()
