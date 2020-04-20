#!/usr/bin/python3

import torch
import torch.nn as nn
import numpy as np

class RNN(nn.Module):
    def __init__(self, inputs=5, categories=4, layers=6):
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



def generate_data(rows=3, columns=4, samples=10, categories=4):
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

    return X, y



def get_data(synthetic=True, categories=4):
    if synthetic:
        return generate_data(categories=4)
    else:
        #TODO: Data should be returned in two lists, namely X and y. Both X and
        #y are list of lists, e.g., X=[[1, 3, 4], [2, 4, 1], ...] and y = [[0], [2],
        #...]. The lists inside X are equally-sized, and every list corresponds to a
        #specific student. Lists inside y are all sized 1 and indicate the label of
        #students based on their final grade
        print('To be implemented ...')
        exit()



if __name__ == "__main__":

    model_parameters = {
        'categories': 4,        # 0: lowest-achiever students, 3: highest-achiever students
        'layers': 6,            # the number of layers of the LSTM model
        'learning_rate': 0.001,
        'epochs': 100
    }

    setup = {
        'loss_report': 20,      # the loss value is printed once every loss_report times
        'train_portion': 1.0,   # the fraction of dataset from the beginning that will be used for training
        'test_portion': 1.0     # the fraction of dataset from the end that will be used for testing
    }

    X, y = get_data(synthetic=True, categories=model_parameters['categories'])

    assert len(X) == len(y)
    rows = len(X[0])
    cols = len(X[0][0])

    print('The dataset includes the record of {} students, each with {} timesteps, where every timestep includes the information of {} activities!'.format(len(X), rows, cols))

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

    print ('Accuracy: {:.3f}'.format(correct_predictions / len(y)))

