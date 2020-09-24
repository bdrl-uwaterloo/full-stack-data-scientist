import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sklearn.metrics import accuracy_score

df = pd.read_csv('datasets_heart.csv')

sns.countplot(x='sex',data=df,palette="muted")
plt.xlabel(" Woman/ Man")

sns.countplot(hue='sex',x='target', data=df,palette="muted")
plt.legend(labels=['Female', 'Male'])

female_data= df [ df['sex']==0] #Only consider female
female_data.info()

feature_lis = ['trestbps','oldpeak','age']
feature_1 = female_data[feature_lis[0]]
feature_2 =  female_data[feature_lis[1]]
feature_3 =    female_data[feature_lis[2]]
target =   female_data['target']

from sklearn.decomposition import PCA

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
pca=PCA(n_components=3,svd_solver='full').fit_transform(df)  
ax.scatter(pca[:,0] , pca[:,1] , pca[:,2], c= df.target.values, cmap = mcolors.ListedColormap(colors))
#ax.view_init(elev=60, azim=-30)
plt.title('PCA')

pca=PCA(n_components=3,svd_solver='full').fit(df)  
print (pca.explained_variance_)
print (pca.explained_variance_ratio_)
print (pca.explained_variance_ratio_.cumsum())


## KPCA - Explained variance ###########################
from sklearn import manifold 
from sklearn.decomposition import KernelPCA

KPCA_Y = KernelPCA(n_components= 3,kernel='poly',degree = 10).fit_transform(df) 
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(KPCA_Y[:,0] , KPCA_Y[:,1] , KPCA_Y[:,2], c= df.target.values, cmap = mcolors.ListedColormap(colors))
plt.title('KPCA')

## Isomap
isomap_Y = manifold.Isomap(n_components = 3, n_neighbors = 10).fit_transform(df)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(MDS_Y [:,0] , MDS_Y [:,1] , MDS_Y [:,2], c= df.target.values, cmap = mcolors.ListedColormap(colors))
plt.title('Isomap')