from read import *
from artificial_neural_network import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')


df = read_data()
ann = ANN(df)
ann.split_test(.3)
epoch_axis, train_error_axis = ann.train_network(ann.x_train, .5, 100, ann.forward_propagate_linear)
plt.plot(epoch_axis, train_error_axis, label='Linear Output Layer')
ann = ANN(df)
ann.split_test(.3)
epoch_axis, train_error_axis = ann.train_network(ann.x_train, .5, 100, ann.forward_propagate_sigmoid)
plt.plot(epoch_axis, train_error_axis, label='Sigmoid Output Layer')
ann = ANN(df)
ann.split_test(.3)
epoch_axis, train_error_axis = ann.train_network(ann.x_train, .5, 100, ann.forward_propagate_softmax)
plt.plot(epoch_axis, train_error_axis, label='Softmax Output Layer')
plt.title('Training Error Using 30% of Data (~1300 emails)')
plt.xlabel('Epoch Number')
plt.ylabel('Relative Squared Error')
plt.legend()
plt.show()

