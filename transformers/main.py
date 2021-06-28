"""
Runs through training and plotting of the loss function for the current running model
"""
from collections import Counter, OrderedDict
import math
import time

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from models.transformer import TransformerModel
from scripts.train_transformer_v1 import (
    batchify,
    conv,
    convert_outpupt_to_tensor,
    data_process,
    evaluate,
    generate_square_subsequent_mask,
    get_batch,
    train,
)
from utils import Config

if __name__ == "__main__":

    # TODO: make this config file be through command line
    cnf = Config.load_toml("options/config.toml")

    wells = pd.read_csv(f"export_csv/{cnf.data.year}.csv", index_col=[1, 0])

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

    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocab = vocab(ordered_dict)

    # Lets break it down using test train val split
    x_train, x_val, x_test = x[:1000:5], x[1000:1250:5], x[1250:3899:5]
    # Process data
    train_data = data_process(x_train.to_numpy(dtype="int64"), tokenizer, vocab)
    val_data = data_process(x_val.to_numpy(dtype="int64"), tokenizer, vocab)
    test_data = data_process(x_test.to_numpy(dtype="int64"), tokenizer, vocab)

    train_data = batchify(train_data, cnf.batch_size, cnf.device)
    val_data = batchify(val_data, cnf.eval_batch_size, cnf.device)
    test_data = batchify(test_data, cnf.eval_batch_size, cnf.device)

    best_val_loss = float("inf")
    best_model = None

    ntokens = len(vocab)  # the size of vocabulary

    # Define our TransformerModel
    model = TransformerModel(
        ntokens,
        cnf.model.d_model,
        cnf.model.nhead,
        cnf.model.nhidden,
        cnf.model.nlayers,
        cnf.model.dropout,
    ).to(cnf.device)

    criterion = nn.MSELoss()
    if cnf.loss == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()

    lr = cnf.lr

    optimizer = None
    if cnf.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if cnf.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    assert optimizer

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 1.0, gamma=0.95  # type:ignore
    )

    # Training loop
    for epoch in range(1, cnf.epochs + 1):
        epoch_start_time = time.time()
        train(model, optimizer, criterion, ntokens, train_data, cnf)
        val_loss = evaluate(model, criterion, ntokens, val_data, cnf)
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
    test_loss = evaluate(best_model, criterion, ntokens, test_data, cnf)

    data_source = test_data
    full_output = []
    src_mask = torch.ones(cnf.input_length, cnf.input_length)
    with torch.no_grad():
        for i in range(0, data_source.size(0), cnf.input_length):
            data, targets = get_batch(data_source, i, cnf)
            if data.size(0) != cnf.input_length:
                src_mask = generate_square_subsequent_mask(data.size(0)).to(cnf.device)

            output = model(data, src_mask)
            full_output.append(output)
    full_output = torch.cat(full_output)
    full_output = convert_output_to_tensor(full_output, vocab)

    print(f"Test Loss: {test_loss} Best Validation Loss: {best_val_loss}")

    # Export results into dataframe (slices are used due to size differences)
    results = pd.DataFrame(
        {
            "Output": [t.item() for t in full_output.view(-1)],
            "Input": conv(test_data.view(-1), vocab)[105:],
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
    plt.savefig(f"results_{cnf.data.year}.png")
    plt.show()
