from read import *
from kNN import *
import matplotlib.pyplot as plt
from uni_tree import *
plt.style.use('ggplot')

# Read data in and clean it
df = read_data()

# Run kNN with each number and graph each result
def display_kNN():
    c_mats, acc, trp, ppv, tnr, f1, error = [], [], [], [], [], [], []
    indices = []
    [indices.append(i) for i in range(2, 9)]
    indices.append(17)
    indices.append(33)
    for i in indices:
        tup = kNN(df, i)
        c_mats.append(tup[0])
        acc.append(tup[1])
        trp.append(tup[2])
        ppv.append(tup[3])
        tnr.append(tup[4])
        f1.append(tup[5])
        error.append(tup[6])
    #plt.plot(indices, acc, label='Accuracy')
    #plt.plot(indices, trp, label='TRP')
    #plt.plot(indices, ppv, label='PPV')
    #plt.plot(indices, tnr, label='TNR')
    #plt.plot(indices, f1, label='F1 Score')
    plt.plot(indices, error, label='Error', c='r', linestyle='dashed', marker='o')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Error')
    plt.title('k-Nearest-Neighbors Error Report')
    plt.show()

#display_kNN()
uni_tree(df)

