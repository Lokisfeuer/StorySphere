from write_advs import *
from rnn import train_model, acc
import autoencoder as autoencoder
import torch


def anna():
    data = all_pre_encodings()
    encoder, decoder = train_model(data)
    torch.save({encoder, f'anna_encoder.pt'})
    torch.save({decoder, f'anna_decoder.pt'})
    return encoder, decoder


def bernd():
    real_encodings = all_real_encodings()
    # split by types.
    bernds = {}
    for unit_type, data_set in real_encodings.items():
        encoder, decoder = train_model(data_set)
        bernds.update({unit_type: data_set})
        torch.save(encoder, f'bernd_encoder_{unit_type}.pt')  # if unit_type is not a string, then take .__name__
        torch.save(decoder, f'bernd_decoder_{unit_type}.pt')
    return bernds


def charlene():
    data = []
    for i in many_small_adventures():
        data.append(i.to_vector(use_autoencoder=False))
    encoder, decoder = autoencoder.train(data)
    torch.save(encoder, f'charlene_encoder.pt')
    torch.save(decoder, f'charlene_decoder.pt')
    # train autoencoder


if __name__ == '__main__':
    # get twitter dataset ready
    # add different unit types and their features
    anna()
    bernd()
    charlene()
