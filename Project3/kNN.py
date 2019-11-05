import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix


def kNN(df, k):
    # Separate values from class values
    X = df.iloc[:, 1:-1].values
    y = df.iloc[:, -1].values

    # 80% training data to 20% learning to avoid over-fitting and standardization
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Neighbor implementation
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    c_matrix = confusion_matrix(y_test, y_pred)
    accuracy = (c_matrix[0][0]+c_matrix[1][1])/(c_matrix[0][0]+c_matrix[1][0]+c_matrix[0][1]+c_matrix[1][1])
    tpr = c_matrix[1][1]/(c_matrix[1][1]+c_matrix[1][0])
    ppv = c_matrix[1][1]/(c_matrix[1][1]+c_matrix[0][1])
    tnr = c_matrix[0][0]/(c_matrix[0][0]+c_matrix[0][1])
    f1_score = 2*ppv*tpr/(ppv+tpr)
    error = np.mean(y_pred != y_test)
    print("Confusion Matrix:\n", c_matrix)
    print("Accuracy:", accuracy)
    print("TPR:", tpr)
    print("PPV:", ppv)
    print("TNR:", tnr)
    print("F1 Score:", f1_score)
    return (c_matrix, accuracy, tpr, ppv, tnr, f1_score, error)
