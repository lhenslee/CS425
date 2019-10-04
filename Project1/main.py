import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Read data from auto-mpg.data into dataframe.
Do mean imputation on horsepower.
Separate data to training, validation, and test sets.
"""
data = pd.read_csv('auto-mpg.data', delim_whitespace=True,
                   names=['mpg', 'cylinders', 'displacement',
                          'horsepower', 'weight', 'acceleration',
                          'model year', 'origin', 'car name'])
hp_mean = 0
hp_count = 0
for hp in data['horsepower']:
    if hp != '?':
        hp_mean += float(hp)
        hp_count += 1
hp_mean /= hp_count
data['horsepower'] = data['horsepower'].replace('?', hp_mean)
data['horsepower'] = data['horsepower'].astype(float)
size = len(data.index)
training = data[0:int(size*.5)]
validation = data[int(size*.5)+1:int(size*.8)]
test = data[int(size*.8)+1:size]


def mlr(parameters, x_values):
    """
    Multivariate linear regression model. Adds together the product of each
    w parameter and each x value. The first x value must be one.
    :param parameters: w_1, w_2, ..., w_d
    :param x_values: x_1, x_2, ..., x_d
    :return: The estimated value given parameters and x values.
    """
    result = 0
    for param, x in zip(parameters, x_values):
        result += param*x
    return result


def get_parameters(data_set, x_variables):
    """
    Find the parameters for multivariate linear/polynomial regression with the given
    data_set using the x_variables.
    :param data_set: Pandas dataframe to learn from.
    :param x_variables: List of d variable names to learn with.
    :return: w_0, w_1, ..., w_d
    """
    rand_var = pd.DataFrame()
    for var in x_variables:
        rand_var[var] = data_set[var]
    rand_var = rand_var.to_numpy()
    rand_var = np.insert(rand_var, 0, 1, axis=1)
    results = data_set['mpg'].to_numpy()
    parameters = np.matmul(np.transpose(rand_var), rand_var)
    parameters = np.linalg.inv(parameters)
    parameters = np.matmul(parameters, np.transpose(rand_var))
    parameters = np.matmul(parameters, results)
    return parameters


def get_estimates(df, parameters, x_variables):
    estimates = []
    for row in df.iterrows():
        x_vals = [1]
        for var in x_variables:
            x_vals.append(row[1][var])
        estimates.append(mlr(parameters, x_vals))
    return estimates


x_vars = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
          'model year', 'origin']
pars = get_parameters(training, x_vars)
y = get_estimates(training, pars, x_vars)
plt.plot(training.index, y, label='Estimate')
plt.plot(training.index, training['mpg'], label='Actual')
plt.legend()
plt.show()
