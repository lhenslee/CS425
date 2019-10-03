import pandas as pd

data = pd.read_csv('auto-mgp.data', delim_whitespace=True)
print(data.head())
