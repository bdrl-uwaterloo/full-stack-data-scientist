
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sklearn.metrics import accuracy_score
# Load in required data from data file


df = pd.read_csv('datasets_heart.csv')

sns.countplot(x='sex',data=df,palette="muted")
plt.xlabel("Female/ Male")

sns.countplot(hue='sex',x='target', data=df,palette="muted")
plt.legend(labels=['Female', 'Male'])

female_data= df [ df['sex']==0] #Only consider female
female_data.info()

feature_lis = ['trestbps','oldpeak','age']
feature_1 = female_data[feature_lis[0]]
feature_2 =  female_data[feature_lis[1]]
feature_3 =    female_data[feature_lis[2]]
target =   female_data['target']

X = female_data[feature_lis]
colors = [ 'forestgreen', 'red']
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=1).fit(X)
y_pred = kmeans.predict(X)

fig5 = plt.figure(5,figsize = (8,5))
ax = fig5.add_subplot(111, projection='3d')
ax.scatter(feature_1, feature_2, feature_3, c= y_pred, cmap = mcolors.ListedColormap(colors))
plt.legend(handles=scatter.legend_elements()[0], labels=['No Heart Desease', 'Heart Desease'])

ax.set_xlabel(feature_lis[0])
ax.set_ylabel(feature_lis[1])
ax.set_zlabel(feature_lis[2])
ax.set_title('K-Means')
plt.show()
print(accuracy_score(target, y_pred))

from sklearn.mixture import GaussianMixture

gmm_clf = GaussianMixture(n_components=2, random_state =1)
gmm_clf.fit(X)


from sklearn.cluster import AgglomerativeClustering
clus = AgglomerativeClustering(n_clusters=2).fit(X)
y_pred = clus.fit_predict(X)
fig5 = plt.figure(5,figsize = (8,5))
ax = fig5.add_subplot(111, projection='3d')
ax.scatter(feature_1, feature_2, feature_3, c= y_pred, cmap = mcolors.ListedColormap(colors))

ax.set_xlabel(feature_lis[0])
ax.set_ylabel(feature_lis[1])
ax.set_zlabel(feature_lis[2])
ax.set_title('Hierarchical Clustering')
plt.show()
print(accuracy_score(target, y_pred))

pca=PCA(n_components=3,svd_solver='full').fit(df)  
print (pca.explained_variance_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())