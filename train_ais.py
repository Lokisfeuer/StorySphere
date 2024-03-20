from write_advs import *
import rnn as rnn
import autoencoder as autoencoder
import torch


def anna():
    data = all_pre_encodings()
    params = {
        'data': data[:100],
        'n_epochs': 2,
        'plot_every': 2,
        'lr': 0.01,
        'hidden_size': PRE_RNN_HIDDEN,  # You can change this variable at the start of adventure.py
        'weight_decay': 0,
        'start_factor': 1.,
        'end_factor': 0.00025,
    }

    train_accuracy, validation_accuracy = rnn.acc(**params)  # takes the same params as train_model
    print(f'Training accuracy: {train_accuracy}, \t\tvalidation accuracy: {validation_accuracy}')
    encoder, decoder = rnn.train_model(**params)
    torch.save(encoder, f'anna_encoder.pt')
    torch.save(decoder, f'anna_decoder.pt')
    return encoder, decoder


def bernd():
    real_encodings = all_real_encodings(n=100)  # n says how many adventures get generated

    # split by types.
    bernds = {}
    for unit_type, data_set in real_encodings.items():
        element_length = len(data_set[0][0])
        for i in range(len(data_set)):
            # add start token.
            data_set[i] = [[1 for _ in range(element_length)]] + data_set[i]
            max_length = 40  # maximum amount of elements per sequence
            # append elements until the standard sequence length is reached
            while len(data_set[i]) < max_length:
                data_set[i].append([0. for _ in range(element_length)])  # element with only zeros to fill sequence.
            assert len(data_set[i]) == max_length

        # split into validation and training data
        random.shuffle(data_set)
        val_len = round(len(data_set) * 0.1)
        train_accuracy, validation_accuracy = rnn.acc(data=data_set[val_len:], n_epochs=2, val_data=data_set[:val_len], max_length=max_length)  # takes the same params as train_model
        print(f'Training accuracy: {train_accuracy}, \t\tvalidation accuracy: {validation_accuracy}')
        encoder, decoder = rnn.train_model(data_set, max_length=max_length)  # train model
        bernds.update({
            unit_type: {'enoder': encoder, 'decoder': decoder}
        })
        torch.save(encoder, f'bernd_encoder_{unit_type}.pt')  # if unit_type is not a string, then take .__name__
        torch.save(decoder, f'bernd_decoder_{unit_type}.pt')  # though unit_type should be a string
    return bernds


def charlene():
    # train autoencoder
    data = []
    train_data = []
    val_data = []
    for i in many_small_adventures(n=20):
        x = i.to_vector(use_autoencoder=False)  # get adventure encoding
        data.append(x)
        '''
        if random.random() < 0.1:
            val_data.append(x)
        else:
            train_data.append(x)
        '''
    train_accuracy, validation_accuracy = autoencoder.acc(data=data, epochs=2, lr=0.0001)
    # acc takes data as list because it still needs to split it data train and eval set.
    print(f'Training accuracy: {train_accuracy}, \t\tvalidation accuracy: {validation_accuracy}')
    data = torch.stack(data)  # dim=0/1?
    encoder, decoder = autoencoder.train_model(data, epochs=2)
    torch.save(encoder, f'charlene_encoder.pt')
    torch.save(decoder, f'charlene_decoder.pt')
    return encoder, decoder


if __name__ == '__main__':
    # TODO:
    #  add different unit types and their features in adventure.py
    #   - include some list of ids features
    #  test

    # anna()
    bernd()
    # charlene()

# for rnn.train_model() you can adjust the following parameters within this file in the train_model() as parameters:
# data, max_length=50, hidden_size=128, batch_size=32, n_epochs=30, print_every=5, plot_every=5, lr=0.001,
# criterion=None
# you can also adjust the optimizers. For that you need to go into the rnn.py file to the train_model function.
# furthermore you might want to adjust the size and amount of the layers. Don't.
# (That is not possible nor useful with rnns.)
# instead adjust hidden_size like the other parameters.

# for autoencoder.train_model() you can adjust the following within this file in the train_model() as parameters:
# data, encoding_size=1024, epochs=20, loss_function=None, lr=0.1, weight_decay=1e-8
# you can adjust the optimizer here as well in the file in the train_model function.
# you can adjust size and amount of the layers in the AutoEncoder class definiton. I believe they are quite good already
# Instead focus on adjusting encoding size; that implicity changes the sizes of the layers as well.
