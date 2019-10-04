import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def get_parameters(data_set, x_variables, target='mpg'):
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
    results = data_set[target].to_numpy()
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


"""
Read data from auto-mpg.data into dataframe.
Do multivariate linear regression imputation on horsepower.
Separate data to training, validation, and test sets.
"""
data = pd.read_csv('auto-mpg.data', delim_whitespace=True,
                   names=['mpg', 'cylinders', 'displacement',
                          'horsepower', 'weight', 'acceleration',
                          'model year', 'origin', 'car name'])
hp_null = []
for row in data.iterrows():
	if row[1]['horsepower'] == '?':
		hp_null.append(row[0])
data1 = data.drop(hp_null)
x_vars = ['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'model year',
		   'origin']
data1['horsepower'] = data1['horsepower'].astype(float)
hp_pars = get_parameters(data1, x_vars, 'horsepower')
for row in data.iterrows():
	if row[1]['horsepower'] == '?':
		x_vals = [1, row[1]['mpg'], row[1]['cylinders'], row[1]['displacement'],
			      row[1]['weight'], row[1]['acceleration'], row[1]['model year'],
				  row[1]['origin']]
		data.loc[row[0], 'horsepower'] = mlr(hp_pars, x_vals)
data['horsepower'] = data['horsepower'].astype(float)
size = len(data.index)
training = data[0:int(size*.5)]
validation = data[int(size*.5)+1:int(size*.8)]
test = data[int(size*.8)+1:size]


x_vars = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration',
          'model year', 'origin']
pars = get_parameters(training, x_vars)
y = get_estimates(training, pars, x_vars)
plt.plot(training.index, y, label='Estimate')
plt.plot(training.index, training['mpg'], label='Actual')
plt.legend()
plt.show()
