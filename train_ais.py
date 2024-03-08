from write_advs import *
import rnn as rnn
import autoencoder as autoencoder
import torch


def anna():
    data = all_pre_encodings()
    encoder, decoder = rnn.train_model(data, n_epochs=2)
    torch.save(encoder, f'anna_encoder.pt')
    torch.save(decoder, f'anna_decoder.pt')
    return encoder, decoder


def bernd():
    real_encodings = all_real_encodings()

    # split by types.
    bernds = {}
    for unit_type, data_set in real_encodings.items():
        element_length = len(data_set[0][0])
        for i in range(len(data_set)):
            # add start token.  # TODO: clean up
            data_set[i] = [[1 for _ in range(element_length)]] + data_set[i]

            # append elements until the standard sequence length is reached
            while len(data_set[i]) < 40:  # todo: parameterize this properly
                data_set[i].append([0. for _ in range(element_length)])  # element with only zeros.
        encoder, decoder = rnn.train_model(data_set)  # train model
        bernds.update({
            unit_type: {'enoder': encoder, 'decoder': decoder}
        })
        torch.save(encoder, f'bernd_encoder_{unit_type}.pt')  # if unit_type is not a string, then take .__name__
        torch.save(decoder, f'bernd_decoder_{unit_type}.pt')  # though unit_type should be a string
    return bernds


def charlene():
    # train autoencoder
    data = []
    for i in many_small_adventures(n=20):
        x = i.to_vector(use_autoencoder=False)  # get adventure encoding
        data.append(x)
    data = torch.stack(data)  # dim=0/1?
    encoder, decoder = autoencoder.train_model(data, epochs=2)
    torch.save(encoder, f'charlene_encoder.pt')
    torch.save(decoder, f'charlene_decoder.pt')


if __name__ == '__main__':
    # TODO:
    # add different unit types and their features in adventure.py
    # test

    anna()
    bernd()
    charlene()
