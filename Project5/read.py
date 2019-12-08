import pandas as pd
from sklearn.preprocessing import StandardScaler
from random import shuffle, seed
import numpy as np


class File:
    scaler = StandardScaler()

    def __init__(self, name, func=None, shape='ovr', ws_delim=False):
        if func is None:
            func = do_nothing
        self.name = name
        self.func = func
        self.shape = shape
        self.ws_delim = ws_delim

    def get_reg_data(self, problem):
        df = pd.read_csv(self.name, header=None, delim_whitespace=self.ws_delim)
        df.loc[:, df.columns[-1]] = df[df.columns[-1]].apply(self.func)
        data = df.to_numpy()
        if problem == 1:
            data = data[:, 3:]
        data[:, :-1] = StandardScaler().fit_transform(data[:, :-1])
        return data

    def get_data(self):
        return pd.read_csv(self.name, header=None, delim_whitespace=self.ws_delim).to_numpy()


class DataSet:
    X_val = None
    y_val = None

    def __init__(self, problem=0, train_size=.5, val_size=.35, seed_no=None):
        self.problem = problem
        if seed_no is not None:
            seed(seed_no)
        files = [
            File('ionosphere.data', reg_ionosphere),
            File('vowel-context.data', shape='ovo', ws_delim=True),
            File('sat.trn', shape='ovo', ws_delim=True),
            File('sat.tst', shape='ovo', ws_delim=True)
        ]
        if problem < 2:
            file = files[problem]
            data = file.get_reg_data(self.problem)
            shuffle(data)
            train_end = int(len(data)*train_size)
            val_end = int(len(data)*(train_size+val_size))
            self.X_train = data[:train_end, :-1]
            self.y_train = data[:train_end, -1]
            self.X_val = data[train_end+1:val_end, :-1]
            self.y_val = data[train_end+1:val_end, -1]
            self.X_test = data[val_end+1:, :-1]
            self.y_test = data[val_end+1:, -1]
            self.shape = file.shape
        else:
            data = files[2].get_reg_data(self.problem)
            self.X_train = data[:, :-1]
            self.y_train = data[:, -1]
            data_test = files[3].get_reg_data(self.problem)
            self.X_test = data_test[:, :-1]
            self.y_test = data_test[:, -1]
            self.shape = files[2].shape
        self.data = data


def do_nothing(x):
    return x


def reg_ionosphere(x):
    return 0 if x == 'b' else 1

