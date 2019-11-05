import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def split_dataset(df):
    # split test and vars
    X = df.iloc[:, :-1].values
    Y = df.iloc[:, -1].values

    # make training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    return X, Y, X_train, X_test, y_train, y_test

def train_using_gini(X_train, X_test, y_train, depth):
    clf_gini = DecisionTreeClassifier(criterion="gini",
                                      random_state=100, max_depth=depth, min_samples_leaf=5)

    clf_gini.fit(X_train, y_train)
    return clf_gini


def train_using_entropy(X_train, X_test, y_train, depth):
    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=depth, min_samples_leaf=5)

    clf_entropy.fit(X_train, y_train)
    return clf_entropy


def prediction(X_test, clf_object):
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred


# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
    print("Confusion Matrix: ",
          confusion_matrix(y_test, y_pred))

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)

    print("Report : ",
          classification_report(y_test, y_pred))


# Driver code
def uni_tree(df):
    # Building Phase
    X, Y, X_train, X_test, y_train, y_test = split_dataset(df)
    error_gini = []
    error_entropy = []
    indices = []
    for i in range(2, 9):
        indices.append(i)
        clf_gini = train_using_gini(X_train, X_test, y_train, i)
        clf_entropy = train_using_entropy(X_train, X_test, y_train, i)

        # Operational Phase
        print("Results Using Gini Index:")

        # Prediction using gini
        y_pred_gini = prediction(X_test, clf_gini)
        error_gini.append(np.mean(y_pred_gini != y_test))
        cal_accuracy(y_test, y_pred_gini)

        print("Results Using Entropy:")
        # Prediction using entropy
        y_pred_entropy = prediction(X_test, clf_entropy)
        error_entropy.append(np.mean(y_pred_entropy!=y_test))
        cal_accuracy(y_test, y_pred_entropy)

    plt.plot(indices, error_gini, label="Error", c='r', linestyle='dashed', marker='o')
    plt.legend()
    plt.xlabel('Tree Depth')
    plt.ylabel('Error')
    plt.title('Gini Style Error Report')
    plt.show()

    plt.plot(indices, error_entropy, label="Error", c='r', linestyle='dashed', marker='o')
    plt.legend()
    plt.xlabel('Tree Depth')
    plt.ylabel('Error')
    plt.title('Entropy Style Error Report')
    plt.show()
