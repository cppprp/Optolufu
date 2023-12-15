import pandas as pd
import sklearn.preprocessing
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

file = '/Users/angelika.svetlove/research files/Optolufu_fibrosis/test/cor_out_av.csv'
df = pd.read_csv(file)
y = df['Tree'].values
x = list(df.columns[2:])
print(y)
#into = pd.read_csv(file, )
X = df[df.columns[2:]]
X=X.set_index(y)
#X = sklearn.preprocessing.MinMaxScaler().fit_transform(X)
#P = PCA()
#X=P.fit_transform(X)

#print (X)
clf = DecisionTreeClassifier()
clf.fit(X, y)
fig = plt.figure(figsize=(18,10))
tree.plot_tree(clf, feature_names=x)
plt.show()
#fig1 = plt.figure(figsize=(18,10))
g = sn.clustermap(X, z_score=True, metric='euclidean', method='average', row_cluster=True, col_cluster=True)
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)  # For y axis
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90) # For x axis
#g.ax_heatmap.set_xticklabels(x, rotation=90)
#g.ax_heatmap.set_yticklabels(y, rotation=0)
plt.tight_layout()
plt.show()
