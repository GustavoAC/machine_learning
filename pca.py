import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

digits = load_digits()
train_img, test_img, train_lbl, test_lbl = train_test_split( digits.data, digits.target, test_size=1/4.0, random_state=0)

scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_img)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

pca = PCA(0.95)
pca.fit(train_img)
# print(pca.n_components_)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)
print(train_img)