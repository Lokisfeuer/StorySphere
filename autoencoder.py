# This file is mediocre clean and mostly copied from here:
# https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/
import random

import torch
import matplotlib.pyplot as plt

import math


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size, encoding_size=1024):
        super().__init__()

        # Encodes inputs in five layers gradually getting smaller to encoding size.
        stepsize = math.ceil((input_size - encoding_size) / 5)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size - stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - stepsize, input_size - 2 * stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - 2 * stepsize, input_size - 3 * stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - 3 * stepsize, input_size - 4 * stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - 4 * stepsize, encoding_size)
            # maybe add a Sigmoid layer here to normalize encodings?
        )

        # reverses encoding process.
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_size, input_size - 4 * stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - 4 * stepsize, input_size - 3 * stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - 3 * stepsize, input_size - 2 * stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - 2 * stepsize, input_size - 1 * stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - 1 * stepsize, input_size)
        )

    def forward(self, x):
        # encode-decode-return
        # useful for training and accuracy testing. Not good for inference (actual usage).
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_model(data, encoding_size=1024, epochs=20, loss_function=None, lr=0.0001, weight_decay=1e-8):
    input_size = len(data[0])
    data = torch.Tensor(data)

    # Model Initialization
    model = AutoEncoder(input_size, encoding_size)

    if loss_function is None:
        # Validation using MSE Loss function
        loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    losses = []
    for epoch in range(epochs):
        for element in data:

            # Output of Autoencoder
            reconstructed = model(element)

            # Calculating the loss function
            loss_function = torch.nn.MSELoss()
            loss = loss_function(reconstructed, element)

            # The gradients are set to zero,
            # the gradient is computed and stored.4
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward(retain_graph=True)  # this solved a bug, but I have no idea what it does.
            # It might even cause the model to not learn. I don't know.
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss.item())

    # losses = [i for i in losses if i < 10000]
    losses = [i if i < 100 else 100. for i in losses]  # clean up giant losses

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    plt.plot(losses[-100:])
    plt.savefig('Charlene-loss.png')  # this function takes looong. why?
    plt.show()  # this doesn't work reliable on my machine.

    return model.encoder, model.decoder


def check(data, encoder, decoder, sample, l):
    # import sklearn as sklearn
    out = encoder(torch.Tensor(sample))
    result = decoder(out)
    real_l = l(result, torch.Tensor(sample)).item()

    for i in data:
        if l(result, torch.FloatTensor(i)).item() < real_l:
            return False
    return True


def acc(data, encoding_size=1024, epochs=20, loss_function=None, lr=0.1, weight_decay=1e-8):
    # this function is untested.
    eval_len = round(len(data) * 0.1)
    random.shuffle(data)
    val_data = data[:eval_len]
    data = data[eval_len:]
    encoder, decoder = train_model(torch.stack(data), encoding_size, epochs, loss_function, lr, weight_decay)
    l = torch.nn.MSELoss()

    # get training accuracy
    good = 0
    for sample, idx in zip(data, range(len(data))):
        data_for_this = []
        for i, j in zip(data, range(len(data))):
            if j != idx:
                data_for_this.append(i)
        if check(data_for_this, encoder, decoder, sample, l):
            good += 1

    # get validation accuracy
    val_good = 0
    for sample, idx in zip(val_data, range(len(val_data))):
        data_for_this = []
        for i, j in zip(val_data, range(len(val_data))):
            if j != idx:
                data_for_this.append(i)
        if check(data_for_this, encoder, decoder, sample, l):
            val_good += 1

    return good / len(data), val_good / len(val_data)


if __name__ == "__main__":
    data = []
    for i in range(100):
        randomlist = random.sample(range(1, 1000), 50)
        data.append(randomlist)
    print(acc(data, epochs=2))
