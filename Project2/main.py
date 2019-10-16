import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def linear_regression(x1, x2):
    """
    Get the w vector for the regression of x1 to predict x2
    :param x1: A column of the data
    :param x2: A column of the data
    :return: A function that takes an x value
    """
    tmp = pd.concat([x1, x2], axis=1).dropna().astype(float)
    A = np.array([[len(tmp.index), tmp[tmp.columns[0]].sum()],
                 [tmp[tmp.columns[0]].sum(), tmp[tmp.columns[0]].apply(lambda x: x**2).sum()]])
    y = np.array([tmp[tmp.columns[1]].sum(),
                  (tmp[tmp.columns[0]]*tmp[tmp.columns[1]]).sum()])
    w = np.matmul(np.linalg.inv(A), y)

    def linear_regressor(x):
        return w[1]*x+w[0]
    return linear_regressor


def make_discrete(x):
    if 'x' in str(x):
        return 1
    elif 'pre clin' in str(x):
        return 2
    else:
        return 0


def dollar_to_float(x):
    if '#N/A' in str(x):
        return np.nan
    elif '$-' in str(x):
        return 0
    elif '$' in str(x):
        return float(str(x).replace(',', '').replace('$', ''))


def make_clean(x):
    if type(x) is float or type(x) is int:
        return x
    if x == '-':
        return np.nan
    if ',' in str(x):
        return str(x).replace(',', '')


def get_clean_data():
    df = pd.read_csv('UTK-peers.csv')[0:57].drop('HBC', axis=1)
    df.loc[:, '2014 Med School'] = df['2014 Med School'].apply(make_discrete)
    df.loc[:, 'Vet School'] = df['Vet School'].apply(make_discrete)
    '''dollar_columns = ['Total E&G Expend', 'E&G / St. FTE', 'State Approp Rev',
                      'Tuition/Fee Rev ', 'Endowment', 'Total Research Expenditures ($000)',
                      'Total Expend', 'Total Revenue', 'Enowment / St. FTE',
                      'Total Research Exp - Med School Exp ($000)',
                      '(State/ Tuit)/ St. FTE', 'Med School Res $',
                      'Academic Support Expenditures', 'Student Services Expenditures',
                      'Endowment Figure', 'Endowment per Student FTE']
    for col in dollar_columns:
        df.loc[:, col] = df[col].apply(dollar_to_float)'''
    for col in df.columns[2:]:
        df.loc[:, col] = df[col].apply(make_clean)
        if '$' in str(df[col][0]) or df[col][0] is None:
            df.loc[:, col] = df[col].apply(dollar_to_float)
        df.loc[:, col] = df[col].astype(float)
    print(df.head())
    return df


data = get_clean_data()
'''func = linear_regression(df[df.columns[6]], df['% UG Pell Grants'])
print(func(3))
df = df.dropna()
plt.scatter(df[df.columns[6]], df['% UG Pell Grants'])
plt.plot(df[df.columns[6]], df[df.columns[6]].apply(func))
plt.show()'''

