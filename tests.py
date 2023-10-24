import pandas as pd
from utils import *

data = pd.read_csv('data/train.csv')
print(data.groupby("thal"))