"""
Training model and forward passes
"""

import math
import time

import torch
from torch import nn


def generate_square_subsequent_mask(sz):
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


def get_batch(source, i, cnf):
    """
    Gets a batch shifted over by shift length
    """
    seq_len = min(cnf.batch_size, len(source) - cnf.forecast_window - i)
    data = source[i : i + seq_len]
    target = source[i + cnf.forecast_window : i + cnf.forecast_window + seq_len].reshape(-1)
    return data, target


def batchify(data, bsz, device):
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


def data_process(raw_text_iter, tokenizer, vocab):
    """
    Converts the data into a latent dimension using the given vocab
    """
    data = [
        torch.tensor(
            [vocab[token] for token in tokenizer(str(item))], dtype=torch.long
        )
        for item in raw_text_iter
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))


def conv(raw_output, vocab):
    """
    Converts the latent dimension into data again the opposite of data_process
    """
    return torch.tensor([int(vocab.get_itos()[x]) for x in raw_output.view(-1)])


def convert_output_to_tensor(output, vocab):
    """
    Converts the latent dimension into data again the opposite of data_process
    This version also accounts for the probability outputs using torch.argmax()
    """
    imm = [
        [vocab.get_itos()[torch.argmax(prob).item()] for prob in result] for result in output
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


def train(model, optimizer, criterion, ntokens, train_data, cnf):
    """
    Trains the model for `epoch` iterations
    """
    model.train()  # Turn on the train mode
    total_loss = 0.0
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(cnf.input_length).to(cnf.device)
    log_interval = 200
    for batch, i in enumerate(range(0, train_data.size(0) - 1, cnf.input_length)):
        # Fetch Data
        data, targets = get_batch(train_data, i, cnf)
        optimizer.zero_grad()

        # Genearte attention mask
        if data.size(0) != cnf.input_length:
            src_mask = generate_square_subsequent_mask(data.size(0)).to(cnf.device)

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
                "{:5d}/{:5d} batches | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    batch,
                    elapsed * 1000 / log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


def evaluate(eval_model, criterion, ntokens, data_source, cnf):
    """
    Evaluates the training loss of the given model
    """
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.0
    src_mask = generate_square_subsequent_mask(cnf.input_length).to(cnf.device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, cnf.input_length):
            data, targets = get_batch(data_source, i, cnf)
            if data.size(0) != cnf.input_length:
                src_mask = generate_square_subsequent_mask(data.size(0)).to(
                    cnf.device
                )
            output = eval_model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)
