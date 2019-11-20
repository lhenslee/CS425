from read import *
from artificial_neural_network import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')


df = read_data()

# Threshold Function stuff
ann = ANN(df)
'''ann.split_test(.5)
epoch_axis, train_error_axis = ann.train_network(ann.x_train, .5, 100, ann.forward_propagate_linear)
print('Threshold Output Function')
ann.plot_set(ann.x_test, ann.forward_propagate_linear)
#plt.plot(epoch_axis, train_error_axis, label='Linear Output Layer')'''

# Sigmoid function stuff
'''ann = ANN(df)
ann.split_test(.7)
epoch_axis, train_error_axis = ann.train_network(ann.x_train, .5, 100, ann.forward_propagate_sigmoid)
print('\n\nSigmoid Output Function')
x_axis, predictions = ann.plot_set(ann.x_test, ann.forward_propagate_sigmoid)
#plt.plot(x_axis, predictions, label='Sigmoid Output Layer')'''

#x_axis = [i for i in range(len(ann.x_test))]
#answers = [row[-1] for i, row in ann.x_test.iterrows()]
#plt.plot(x_axis, answers, label='The Actual Outputs')

# Soft max function stuff
'''ann = ANN(df)
ann.split_test(.5)
epoch_axis, train_error_axis = ann.train_network(ann.x_train, .2, 50, ann.forward_propagate_softmax)
print('\n\nSoft Output Function')
ann.plot_set(ann.x_test, ann.forward_propagate_softmax)
print('\n\n')'''

# Graphing stuff
'''plt.plot(epoch_axis, train_error_axis, label='Softmax Output Layer')
plt.title('Test Data and ANN Predictions 50% of Data (~2050 emails)')
plt.xlabel('Index')
plt.ylabel('Classification 1 = Spam, 0 = Not Spam')
plt.legend()
plt.show()'''

