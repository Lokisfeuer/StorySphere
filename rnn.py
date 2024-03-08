import pandas as pd
import random as random

# https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#the-seq2seq-model
# from __future__ import unicode_literals, print_function, division
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, Dataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import math


# TODO: Deal more intelligently with global variables.
plt.switch_backend('agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_token = 0
EOS_token = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = 40  # change this only when seq2seq.py is not used anymore


class CustomDataset(Dataset):
    def __init__(self, data):
        # data is a list of sequences
        # A sequence is a list of elements
        # element is a list of numbers that encode the elements content.
        # the model gets trained to autoencode sequences. It uses an RNN with one cell-run per element.
        # The RNN encoding cell takes an element in each cell-run.
        # the elements need to be normalized and prepared beforehand.
        self.length = len(data)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.data[idx])


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        # self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        # input is a batch of sequences
        # sequence is a list of elements which each consists of multiple numbers.
        # embedded = self.dropout(input)  # this does weird stuff
        output, hidden = self.gru(input)
        return output, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        # input size equals output size
        self.gru = nn.GRU(output_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.element_size = output_size

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        if target_tensor is not None:
            batch_size = encoder_outputs.size(0)
            decoder_input = torch.empty(batch_size, 1, self.element_size, dtype=torch.float,
                                        device=device)  # TODO: find a good start token
            decoder_input = torch.zeros(batch_size, 1, self.element_size, dtype=torch.float, device=device)
        else:
            decoder_input = torch.zeros(1, self.element_size, dtype=torch.float, device=device)

        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                # decoder_input = topi.squeeze(-1).detach()  # detach from history as input
                decoder_input = torch.FloatTensor(decoder_output).detach()
        if target_tensor is not None:
            decoder_outputs = torch.cat(decoder_outputs, dim=1)
        else:
            decoder_outputs = torch.cat(decoder_outputs, dim=0)
        # decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None  # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.out(output)
        return output, hidden


def train_epoch(dataloader, encoder, decoder, encoder_optimizer,
                decoder_optimizer, criterion):
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)

        # to_l1 = decoder_outputs.view(-1, decoder_outputs.size(-1))
        # to_l2 = target_tensor.view(-1)

        loss = criterion(
            decoder_outputs.view(-1),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def train(train_dataloader, encoder, decoder, n_epochs, lr=0.001,
          print_every=100, plot_every=100, encoder_optimizer=None, decoder_optimizer=None, criterion=None):
    if criterion is None:
        criterion = nn.MSELoss()
    if encoder_optimizer is None:
        encoder_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    if decoder_optimizer is None:
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=lr)

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                         epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig("enc_adv_graph.png")
    print('rnn.py finished and saved .png file.')


def to_sequence(data):
    # reuse some elements to form new sequences to generate more data. Then there is enough data.
    seq = []
    clear_data = []
    element_length = len(data[0])
    for i in range(len(data)):
        seq.append(data[i])
        if len(seq) == 10 or random.random() > 0.9:
            while len(seq) < 40:  # todo: parameterize this properly
                seq.append([0. for _ in range(element_length)])
            clear_data.append(seq)
            seq = []
    return clear_data


def train_model(data, hidden_size=128, batch_size=32, n_epochs=30, print_every=5, plot_every=5, lr=0.001, encoder_optimizer=None, decoder_optimizer=None, criterion=None):
    # this top part is not tested.
    # what about .csv files?
    if isinstance(data, str):
        if data.endswith('.npy'):
            data = np.load(data)
            data = data.tolist()
        elif data.endswith('.pickle'):
            import pickle
            with open(data, 'rb') as f:
                data = pickle.load(f)
        elif data.endswith('.json'):
            import json
            with open(data, 'r') as f:
                data = json.load(f)
    if not isinstance(data[0][0], list):
        data = to_sequence(data)
    else:
        pass  # check for lengths and if necessary add zeros until all sequences are length = MAX_LENGTH
    # data is a list of sequences
    # A sequence is a list of elements
    # An element is a list of numbers that encode the elements content.
    # the model gets trained to auto encode sequences. It uses an RNN with one cell-run per element.
    # The RNN encoding cell takes an element in each cell-run.
    # the elements need to be normalized and prepared beforehand.
    element_length = len(data[0][0])
    data = CustomDataset(data)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    encoder = EncoderRNN(element_length, hidden_size).to(device)
    decoder = DecoderRNN(hidden_size, element_length).to(device)

    train(dataloader, encoder, decoder, n_epochs=n_epochs, print_every=print_every, plot_every=plot_every, lr=lr, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer, criterion=criterion)

    return encoder, decoder


def check(data, encoder, decoder, sequence, l):
    # import sklearn as sklearn
    out, hid = encoder(torch.FloatTensor(sequence))
    result, _, _ = decoder(out, hid)
    real_l = l(result, torch.FloatTensor(sequence)).item()
    while sequence in data:
        data.remove(sequence)
    for i in data:
        if l(result, torch.FloatTensor(i)).item() < real_l:
            return False
    return True


def scramble_data(data, n=3):
    seqs = []
    for i in range(n):
        seqs = seqs + to_sequence(data)
        random.shuffle(data)
    return seqs, to_sequence(data)


# TODO: test this function
def acc(train_parameters):
    if 'data' not in train_parameters.keys() or 'val_data' not in train_parameters.keys():
        data = 'allObjectsTwitterEncoded.npy'
        data = np.load(data).tolist()
        data, val_data = scramble_data(data)
    else:
        data = train_parameters['data']
        val_data = train_parameters['val_data']
    possible_train_parameters = ['hidden_size', 'batch_size', 'n_epochs', 'print_every', 'plot_every', 'lr', 'encoder_optimizer', 'decoder_optimizer', 'criterion']
    for i in train_parameters.keys():
        if i not in possible_train_parameters:
            del train_parameters[i]
    encoder, decoder = train_model(data=data, **train_parameters)
    l = torch.nn.MSELoss()
    good = 0
    for sequence in data:
        if check(data, encoder, decoder, sequence, l):
            good += 1
    val_good = 0
    for sequence in val_data:
        if check(val_data, encoder, decoder, sequence, l):
            good += 1
    return good / len(data), val_good / len(val_data)
    # choose or generate random sequence
    # encode-decode it
    # compare result with all sequences
    # see how often right one is closest => calculate accuracy


if __name__ == '__main__':
    print(acc())
    # TODO use BCE to reduce error!
    # train_model(data='allObjectsTwitterEncoded.npy', n_epochs=30)
