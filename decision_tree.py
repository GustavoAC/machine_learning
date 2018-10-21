import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

train_set = pd.read_csv('./dados/optdigits.tra', header = None)

print train_set.corr()[64]