import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def linear_regression(x1, x2):
    """
    Get the w vector for the regression of x1 to predict x2
    :param x1: A column of the data
    :param x2: A column of the result data
    :return: A function that maps an x value to its linear regressor
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
    try:
        return float(x)
    except Exception as e:
        if type(x) is float or type(x) is int:
            return x
        if x == '-':
            return np.nan
        if ',' in str(x):
            return str(x).replace(',', '')


def get_best_corr(corr, col):
    """
    Find the index that correlates most to col in the corr matrix.
    :param corr: A correlation df
    :param col: The column to find max correlation to
    :return: The string that represents the most correlating input
    """
    max_corr = 0
    best_corr = str()
    for index, row in corr.iterrows():
        if abs(row[col]) > max_corr and index != col:
            max_corr = abs(row[col])
            best_corr = index
    return best_corr


def fill_na(df):
    """
    Fill the null values with linear regression on the highest correlated
    value.
    :param df: The df with null values
    :return: The df with filled in null values
    """
    corr = df[df.columns[2:]].astype(float).corr()
    for col in df.columns[2:]:
        best_corr = get_best_corr(corr, col)
        func = linear_regression(df[best_corr], df[col])
        for index, row in df.iterrows():
            if pd.isna(row[col]):
                replacement = func(row[best_corr])
                df.loc[index, col] = func(row[best_corr])
                while pd.isna(replacement):
                    best_corr = get_best_corr(corr.drop(best_corr, axis=0), col)
                    func = linear_regression(df[best_corr], df[col])
                    replacement = func(row[best_corr])
                    df.loc[index, col] = replacement
    return df


def get_clean_data():
    df = pd.read_csv('UTK-peers.csv')[0:57].drop('HBC', axis=1)
    df.loc[:, '2014 Med School'] = df['2014 Med School'].apply(make_discrete)
    df.loc[:, 'Vet School'] = df['Vet School'].apply(make_discrete)
    for col in df.columns[2:]:
        df.loc[:, col] = df[col].apply(make_clean)
        if '$' in str(df[col][0]) or df[col][0] is None:
            df.loc[:, col] = df[col].apply(dollar_to_float)
    df = fill_na(df)
    return df


data = get_clean_data()
data.to_csv('clean.csv')
'''func = linear_regression(data['Med School Res $'], best_corr)
print(func(3))
#data = data.dropna()
plt.scatter(data['Med School Res $'], best_corr)
plt.plot(data['Med School Res $'], data['Med School Res $'].apply(func))
plt.show()'''

