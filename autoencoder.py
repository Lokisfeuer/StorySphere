# This file is mediocral clean and mostly copied from here:
# https://www.geeksforgeeks.org/implementing-an-autoencoder-in-pytorch/
import torch
import matplotlib.pyplot as plt

import math


# Creating a PyTorch class
# 28*28 ==> 9 ==> 28*28
class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size, encoding_size=1024):
        super().__init__()

        # Building a linear encoder with Linear
        # layer followed by Relu activation function
        # 784 ==> 9
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
        )

        # Building a linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 9 ==> 784
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(encoding_size, input_size - 4 * stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - 4 * stepsize, input_size - 3 * stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - 3 * stepsize, input_size - 2 * stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - 2 * stepsize, input_size - 1 * stepsize),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size - 1 * stepsize, input_size),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train(data, encoding_size=1024, epochs=20, loss_function=None, lr=0.1, weight_decay=1e-8):
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
            loss = loss_function(reconstructed, element)

            # The gradients are set to zero,
            # the gradient is computed and stored.4
            # .step() performs parameter update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Storing the losses in a list for plotting
            losses.append(loss)

    # Defining the Plot Style
    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    # Plotting the last 100 values
    plt.plot(losses[-100:])

    return model.encoder, model.decoder
