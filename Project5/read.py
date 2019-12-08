import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


class DataSet:
    scaler = StandardScaler()

    def __init__(self, problem=0, test_size=.33):
        self.problem = problem

        # Read the data in, split to train and test, and standardize data
        if problem == 0:
            self.data = pd.read_csv('ionosphere.data', header=None, delim_whitespace=False)
            self.data.loc[:, self.data.columns[-1]] = self.data[self.data.columns[-1]].apply(reg_ionosphere)
            self.data = self.data.to_numpy()
            self.X_train, self.y_train = self.data[:200, :-1], self.data[:200, -1]
            self.X_test, self.y_test = self.data[200:, :-1], self.data[200:, -1]
        elif problem == 1:
            self.data = pd.read_csv('vowel-context.data', header=None, delim_whitespace=True)
            index_names = self.data[self.data[self.data.columns[2]] == 1]
            train = self.data.drop(index_names).to_numpy()[:, 3:]
            index_names = self.data[self.data[self.data.columns[2]] == 0]
            test = self.data.drop(index_names).to_numpy()[:, 3:]
            self.X_train, self.y_train = train[:, :-1], train[:, -1]
            self.X_test, self.y_test = test[:, :-1], test[:, -1]
            self.data = self.data.to_numpy()[:, 3:]
        elif problem == 2:
            train = pd.read_csv('sat.trn', header=None, delim_whitespace=True).to_numpy()
            test = pd.read_csv('sat.tst', header=None, delim_whitespace=True).to_numpy()
            self.X_train, self.y_train = train[:, :-1], train[:, -1]
            self.scaler.fit(self.X_train)
            self.X_test, self.y_test = test[:, :-1], test[:, -1]
            self.data = np.concatenate(([train, test]))
        self.scaler.fit(self.X_train)
        self.data[:, :-1] = self.scaler.transform(self.data[:, :-1])
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)


def reg_ionosphere(x):
    return 0 if x == 'b' else 1

