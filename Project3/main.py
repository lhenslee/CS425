import pandas as pd

columns = ['ID', 'Clump Thickness', 'Cell Size Uniformity',
           'Cell Shape Uniformity', 'Marginal Adhesion',
           'Single Epithelial Cell Size', 'Bare Nuclei',
           'Bland Chromatin', 'Normal Nucleoli', 'Mitoses',
           'Class']
df = pd.read_csv('breast-cancer-wisconsin.data', header=None, names=columns)

print(df.head())
