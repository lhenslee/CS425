from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import pandas as pd


class SVC:
    clf = svm.SVC(gamma='scale', cache_size=1000)
    problem_names = ['Ionosphere', 'Vowel Context', 'SAT']

    def __init__(self, data):
        self.data = data
        self.problem = data.problem
        self.X_train = data.X_train
        self.y_train = data.y_train
        self.X_test = data.X_test
        self.y_test = data.y_test

    def fit_clf(self):
        # Perform a course grid search to find the best penalty and kernel scale
        param_grid = [
            {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [1, 10, 100, 1000]}
        ]
        clf = self.clf
        self.clf = GridSearchCV(clf, param_grid, cv=5, iid=False)
        self.clf.fit(self.X_train, self.y_train)
        # Perform fine grid search with the best penalty and kernel scale
        param_grid = {
            'C': [self.clf.best_params_['C']], 'kernel': [self.clf.best_params_['kernel']],
            'gamma': ['auto', 'scale', .001, .0001], 'shrinking': [True, False],
            'probability': [True, False], 'decision_function_shape': ['ovo', 'ovr'],
        }
        if self.clf.best_params_['kernel'] == 'poly':
            param_grid['degree'] = ['2, 3, 4, 5']
        if self.clf.best_params_['kernel'] == 'poly' or self.clf.best_params_['kernel'] == 'sigmoid':
            param_grid['coef0'] = 0, .5, 1
        self.clf = GridSearchCV(clf, param_grid, cv=5, iid=False)
        self.clf.fit(self.X_train, self.y_train)
        print('Best Parameters', self.clf.best_params_)

    def report_c_mat(self, set='validation'):
        if set == 'training':
            X = self.data.X_train
            y = self.data.y_train
        else:
            X = self.data.X_test
            y = self.data.y_test
        y_pred = self.clf.predict(X)
        c_mat = confusion_matrix(y, y_pred)
        correct = 0
        total = 0
        for i in range(len(c_mat)):
            correct += c_mat[i][i]
            for j in range(len(c_mat[i])):
                total += c_mat[i][j]
        print('Confusion matrix for', set, 'set on', self.problem_names[self.problem], 'data.')
        print(c_mat)
        print('Accuracy:', correct/total)







