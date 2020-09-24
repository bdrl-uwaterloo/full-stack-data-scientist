# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# %%
import os
os.chdir('../../Data')
df = pd.read_csv('datasets_heart.csv')


# %%
df.info()


# %%
df.head()


# %%
sns.countplot(x='sex',data=df,palette="muted")
plt.xlabel(" Female/ Male")


# %%
sns.countplot(hue='sex',x='target', data=df,palette="muted")
plt.legend(labels=['Female', 'Male'])


# %%
female_data= df [ df['sex']==0] #Only consider female
female_data.info()


# %%
feature_lis = ['trestbps','oldpeak','age']

feature_1 = female_data[feature_lis[0]]
feature_2 =  female_data[feature_lis[1]]
feature_3 =    female_data[feature_lis[2]]
target =   female_data['target']
target


# %%
import matplotlib.colors as mcolors
colors = [ 'forestgreen', 'red']
scatter = plt.scatter(feature_1, feature_2, c= target, cmap = mcolors.ListedColormap(colors))
plt.xlabel(feature_lis[0])
plt.ylabel(feature_lis[1])
plt.legend(handles=scatter.legend_elements()[0], labels=['heart desease', 'no heart desease'])
plt.show()


# %%
from mpl_toolkits.mplot3d import Axes3D
fig2 = plt.figure(2,figsize = (8,5))
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(feature_1, feature_2, feature_3, c= target, cmap = mcolors.ListedColormap(colors))
plt.legend(handles=scatter.legend_elements()[0], labels=['No Heart Desease', 'Heart Desease'])
ax.set_xlabel(feature_lis[0])
ax.set_ylabel(feature_lis[1])
ax.set_zlabel(feature_lis[2])
plt.show()


# %%
X = female_data[feature_lis]


# %%
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X, target)


# %%
y_pred = knn_clf.predict(X)
fig3 = plt.figure(10,figsize = (8,5))
ax = fig3.add_subplot(111, projection='3d')
ax.scatter(feature_1, feature_2, feature_3, c= y_pred, cmap = mcolors.ListedColormap(colors))
plt.legend(handles=scatter.legend_elements()[0], labels=['No Heart Desease', 'Heart Desease'])

ax.set_xlabel(feature_lis[0])
ax.set_ylabel(feature_lis[1])
ax.set_zlabel(feature_lis[2])
ax.set_title('KNN')
plt.show()


# %%
from sklearn.metrics import accuracy_score
accuracy_score(target, y_pred)


# %%
from sklearn.mixture import GaussianMixture

gmm_clf = GaussianMixture(n_components=2, random_state =1)
gmm_clf.fit(X)


# %%
y_pred = gmm_clf.predict(X)
fig3 = plt.figure(3,figsize = (8,5))
ax = fig3.add_subplot(111, projection='3d')
ax.scatter(feature_1, feature_2, feature_3, c= y_pred, cmap = mcolors.ListedColormap(colors))
plt.legend(handles=scatter.legend_elements()[0], labels=['No Heart Desease', 'Heart Desease'])

ax.set_xlabel(feature_lis[0])
ax.set_ylabel(feature_lis[1])
ax.set_zlabel(feature_lis[2])
ax.set_title('GMM')
plt.show()
print(accuracy_score(target, y_pred))


# %%
from sklearn.ensemble import IsolationForest

isofo_clf = IsolationForest( contamination = 0.08)

isofo_clf.fit(X)
y_pred = isofo_clf.predict(X)


# %%
fig4 = plt.figure(4,figsize = (8,5))
ax = fig4.add_subplot(111, projection='3d')
ax.scatter(feature_1, feature_2, feature_3, c= y_pred, cmap = mcolors.ListedColormap(colors))
plt.legend(handles=scatter.legend_elements()[0], labels=['No Heart Desease', 'Heart Desease'])

ax.set_xlabel(feature_lis[0])
ax.set_ylabel(feature_lis[1])
ax.set_zlabel(feature_lis[2])
ax.set_title('Isolation Forest')
plt.show()
print(accuracy_score(target, y_pred))


# %%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=1).fit(X)
y_pred = kmeans.predict(X)


# %%
fig5 = plt.figure(5,figsize = (8,5))
ax = fig5.add_subplot(111, projection='3d')
ax.scatter(feature_1, feature_2, feature_3, c= y_pred, cmap = mcolors.ListedColormap(colors))
plt.legend(handles=scatter.legend_elements()[0], labels=['No Heart Desease', 'Heart Desease'])
y_pred = isofo_clf.predict(X)
ax.set_xlabel(feature_lis[0])
ax.set_ylabel(feature_lis[1])
ax.set_zlabel(feature_lis[2])
ax.set_title('K-Means')
plt.show()
print(accuracy_score(target, y_pred))


# %%
from sklearn.cluster import AgglomerativeClustering
clus = AgglomerativeClustering(n_clusters=2).fit(X)
y_pred = clus.fit_predict(X)
fig5 = plt.figure(5,figsize = (8,5))
ax = fig5.add_subplot(111, projection='3d')
colors = ['forestgreen','red']

ax.scatter(feature_1, feature_2, feature_3, c= y_pred, cmap = mcolors.ListedColormap(colors))
plt.legend(handles=scatter.legend_elements()[0], labels=['No Heart Desease', 'Heart Desease'])

ax.set_xlabel(feature_lis[0])
ax.set_ylabel(feature_lis[1])
ax.set_zlabel(feature_lis[2])
ax.set_title('Hierarchical Clustering')
plt.show()
print(accuracy_score(target, y_pred))


# %%
from sklearn.decomposition import PCA

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
pca=PCA(n_components=3,svd_solver='full').fit_transform(df)  
ax.scatter(pca[:,0] , pca[:,1] , pca[:,2], c= df.target.values, cmap = mcolors.ListedColormap(colors))
#ax.view_init(elev=60, azim=-30)
plt.title('PCA')


# %%
pca=PCA(n_components=3,svd_solver='full').fit(df)  
print (pca.explained_variance_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())


# %%
from sklearn import manifold 
from sklearn.decomposition import KernelPCA

pca=PCA(n_components=3).fit_transform(df)
MDS_Y = manifold.MDS(n_components= 3, max_iter=100, n_init = 2).fit_transform(df)

KPCA_Y = KernelPCA(n_components= 3,kernel='poly',degree = 10).fit_transform(df)


# %%
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
 
colors = ['red', 'forestgreen']

ax.scatter(KPCA_Y[:,0] , KPCA_Y[:,1] , KPCA_Y[:,2], c= df.target.values, cmap = mcolors.ListedColormap(colors))
plt.title('KPCA')
#plt.savefig('3DM_gmm_kpca_poly.png', dpi=72, bbox_inches='tight')


# %%
isomap_Y = manifold.Isomap(n_components = 3, n_neighbors = 10).fit_transform(df)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(MDS_Y [:,0] , MDS_Y [:,1] , MDS_Y [:,2], c= df.target.values, cmap = mcolors.ListedColormap(colors))
plt.title('Isomap')


