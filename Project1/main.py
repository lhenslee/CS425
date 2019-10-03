import pandas as pd

data = pd.read_csv('auto-mpg.data', delim_whitespace=True)
size = len(data.index)
training = data[0:int(size*.5)]
validation = data[int(size*.5)+1:int(size*.8)]
test = data[int(size*.8)+1:size]
print(training)
