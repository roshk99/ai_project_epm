#!/usr/bin/python3

import torch
import torch.nn as nn
import numpy as np
from import_data import *
import time

class RNN(nn.Module):
    def __init__(self, inputs=5, categories=2, layers=6):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size = inputs,
            hidden_size = categories + 1,
            num_layers = layers,
            batch_first = True,
        )

    def forward(self, x):
        out, (h_n, h_c) = self.rnn(x, None)
        return out[:, -1, :]	# Return output at last time-step



def generate_data(rows=3, columns=4, samples=10, categories=2):
    X = []
    y = []

    np.random.seed(0)

    for i in range(samples):
        data_set = []
        for j in range(rows): 
            data = []
            for k in range(columns):
                data.append(np.random.rand())
            data_set.append(data)
        X.append(data_set)
        y.append([np.random.randint(0, high=categories)])

    seq_lengths = [len(x) for x in X]

    return X, y, seq_lengths



def get_data(synthetic=True, categories=2):
    if synthetic:
        return generate_data(categories=2)
    else:
        return import_data(categories)



if __name__ == "__main__":
    start_time = time.time()

    model_parameters = {
        'categories': 2,        # 0: lowest-achiever students, 3: highest-achiever students
        'layers': 10,            # the number of layers of the LSTM model
        'learning_rate': 0.001,
        'epochs': 15
    }

    setup = {
        'loss_report': 1,      # the loss value is printed once every loss_report times
        'train_portion': 0.75,   # the fraction of dataset from the beginning that will be used for training
        'test_portion': 0.25,     # the fraction of dataset from the end that will be used for testing
        'neglect_activity': 1                   # which activity should be removed from the 8 activites
    }

    _X, y, seq_lengths = get_data(synthetic=False, categories=model_parameters['categories'])

    assert len(_X) == len(y)
    rows = len(_X[0])
    cols = len(_X[0][0])

    X = []

    ### Remove activity _i
    _i = setup['neglect_activity']
    assert _i <= 7 and _i >= 0
    for item in _X:
        assert len(item.shape) == 2    #2D Array
        X.append(np.delete(item, _i, 1))

    assert len(X) == len(_X)
    assert len(X[0]) == rows
    assert len(X[0][0]) == cols - 1
    cols -= 1
    
    print("--- %s seconds (time to remove feature) ---" % (time.time() - start_time))

    print('The dataset includes the record of {} students, each with {} timesteps, where every timestep includes the information of {} activities!'.format(len(X), rows, cols))

    print('model_parameters:')
    print(model_parameters)
    print('setup:')
    print(setup)

    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    split = int(setup['train_portion'] * len(X))
    X_train = X[:split]
    y_train = y[:split]

    split = int(len(X) - setup['test_portion'] * len(X))
    X_test = X[split:]
    y_test = y[split:]

    rnn = RNN(cols, model_parameters['categories'], model_parameters['layers'])
    optimizer = torch.optim.Adam(rnn.parameters(), lr=model_parameters['learning_rate'])
    loss_func = nn.CrossEntropyLoss()

    for i in range(model_parameters['epochs']):
        for j, item in enumerate(X_train):
            item = item.unsqueeze(0)
            output = rnn(item)
            loss = loss_func(output, y_train[j])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % setup['loss_report'] == 0:
            print('Loss: ', np.average(loss.detach()))

    correct_predictions = 0
    for i, item in enumerate(X_test):
        #print(y[i])

        outp = rnn(item.unsqueeze(0))
        #print(np.argmax(outp.detach()))

        if int(y_test[i]) == int(np.argmax(outp.detach())):
            correct_predictions += 1

    print ('Accuracy: {:.3f}'.format(correct_predictions / len(y_test)))
    print("--- %s seconds ---" % (time.time() - start_time))
