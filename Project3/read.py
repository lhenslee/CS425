import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def read_data():
    columns = ['ID', 'Clump Thickness', 'Cell Size Uniformity',
               'Cell Shape Uniformity', 'Marginal Adhesion',
               'Single Epithelial Cell Size', 'Bare Nuclei',
               'Bland Chromatin', 'Normal Nucleoli', 'Mitoses',
               'Class']
    df = pd.read_csv('breast-cancer-wisconsin.data', header=None, names=columns)
    df = df.replace('?', np.nan).astype(float)

    # Find highest correlating variable and do linear regression on null values
    corr = df.corr()
    cdf = df.dropna()
    for index, row in df.iterrows():
        for col in columns:
            if str(row[col]) == 'nan':
                most_cor = 0
                cor_name = str()
                for cor in columns:
                    if cor != col and corr[col][cor] > most_cor:
                        cor_name = cor
                        most_cor = corr[col][cor]
                x = cdf[cor].to_numpy().reshape((1, -1))
                y = cdf[col].to_numpy().reshape((1, -1))
                model = LinearRegression().fit(x, y)
                pred = model.predict(x)[0, index]
                df.loc[index, col] = pred

    return df
