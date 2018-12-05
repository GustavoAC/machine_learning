import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

train_set = pd.read_csv('./dados/optdigits.tra', header = None)
test_set = pd.read_csv('./dados/optdigits.tes', header = None)

raw_data = train_set.values[:, :-1]
test_data = test_set.values[:, :-1]

data = np.concatenate((raw_data, test_data), axis=0)
classes = np.concatenate((train_set.values[:,64], test_set.values[:, 64]), axis=0)

x = []
y = []

lda = LinearDiscriminantAnalysis()
lda.fit(data, classes)

data = lda.transform(data)

for i in range(6, 15): 
    cluster = AgglomerativeClustering(n_clusters=i)
    cluster = cluster.fit(data)
    score = davies_bouldin_score(data, cluster.labels_)
    print "num of clusters: ", i, ", score: ", score
    x.append(i)
    y.append(score)

plt.plot(x, y)
plt.xlabel("Number of clusters")
plt.ylabel("DB Index")
plt.show()

