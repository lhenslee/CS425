from random import random
from math import exp
import matplotlib.pyplot as plt


# Find dot product of weights and inputs
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
    return activation


# Get the slope
def transfer_derivative(output):
    return output * (1.0 - output)


# Sigmoid function
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


class ANN:
    x_train = None
    x_test = None
    network = None
    past_l_rate = None
    last_e_rate = None

    def __init__(self, df, d=None, K=2):
        self.df = df
        if d is None:
            d = len(df.columns)-1
        self.d = d
        self.K = K
        self.attributes = df.columns[0:d]
        self.normalize()
        self.split_test()
        #self.d_reduce(2)
        self.initialize_network()

    def normalize(self, df=None):
        if df is None:
            df = self.df
        else:
            self.df = df
        inputs = self.df[self.attributes]
        self.df.loc[:, self.attributes] = (inputs-inputs.mean())/inputs.std()

    def split_test(self, training=.7):
        self.x_train = self.df.loc[:len(self.df)*training, :]
        self.x_test = self.df.loc[len(self.df)*training:, :]
        return self.x_train, self.x_test

    def initialize_network(self, d=None, n_hidden=None, K=None):
        if d is None:
            d = self.d
        if n_hidden is None:
            n_hidden = 2
        if K is None:
            K = self.K
        self.network = list()
        hidden_layer = [{'weights': [random()/10 for i in range(d+1)]} for i in range(n_hidden)]
        self.network.append(hidden_layer)
        output_layer = [{'weights': [random()/10 for i in range(n_hidden+1)]} for i in range(K)]
        self.network.append(output_layer)
        return self.network

    def forward_propagate_linear(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            best_a = activate(layer[0]['weights'], inputs)
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                if activation > best_a:
                    best_a = activation
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                if activation < best_a:
                    neuron['output'] = 1
                else:
                    neuron['output'] = 0
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def forward_propagate_sigmoid(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                neuron['output'] = transfer(activation)
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def forward_propagate_softmax(self, row):
        inputs = row
        for layer in self.network:
            new_inputs = []
            activations = []
            softmax = 0
            for neuron in layer:
                activation = activate(neuron['weights'], inputs)
                activations.append(activation)
                softmax += exp(activation)
            for neuron, activation in zip(layer, activations):
                neuron['output'] = exp(activation)/softmax
                new_inputs.append(neuron['output'])
            inputs = new_inputs
        return inputs

    def backward_propagate_error(self, expected):
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            errors = list()
            if i != len(self.network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for neuron in self.network[i+1]:
                        error += (neuron['weights'][j] * neuron['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    neuron = layer[j]
                    errors.append(expected[j]-neuron['output'])
            for j in range(len(layer)):
                neuron = layer[j]
                neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

    def update_weights(self, row, l_rate):
        for i in range(len(self.network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [neuron['output'] for neuron in self.network[i-1]]
            for neuron in self.network[i]:
                for j in range(len(inputs)):
                    neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
                neuron['weights'][-1] += l_rate * neuron['delta']

    def train_network(self, train, l_rate, n_epoch, f_prop):
        n_outputs = self.K
        epoch_axis = [i for i in range(n_epoch)]
        train_error_axis = []
        for epoch in range(n_epoch):
            sum_error = 0
            for i, row in train.iterrows():
                outputs = f_prop(row.values)
                expected = [0 for i in range(n_outputs)]
                expected[int(row[train.columns[-1]])] = 1
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.backward_propagate_error(expected)
                self.update_weights(row, l_rate)
            print('epoch=%d, learning rate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
            train_error_axis.append(sum_error)
        return epoch_axis, train_error_axis

    def d_reduce(self, size):
        corr = self.df.corr()
        top_corr = ['spam']
        last_del = 'spam'
        corr = corr.drop(last_del, axis=0)
        for i in range(size):
            if abs(corr['spam'].min()) > corr['spam'].max():
                last_del = corr['spam'].idxmin()
                corr = corr.drop(last_del, axis=0)
            else:
                last_del = corr['spam'].idxmax()
                corr = corr.drop(last_del, axis=0)
            top_corr.insert(0, last_del)
        self.df = self.df.loc[:, top_corr]
        self.d = size+1

    def predict(self, row):
        outputs = self.forward_propagate(row)
        return outputs.index(max(outputs))
