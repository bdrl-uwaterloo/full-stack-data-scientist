
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


## Cluster based on three feature
feature_A =   ASRS.iloc[:, :9].sum(axis=1).values
feature_B = ASRS.iloc[:, 9 :].sum(axis=1).values
feature_C = AQ.sum(axis =1).values

## Visualization 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
colors = [ 'slateblue', 'forestgreen','goldenrod']
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(feature_A, feature_B, feature_C, c='b', marker='o')
ax.set_xlabel('Inattentive')
ax.set_ylabel('Hyperactive/Impulsive')
ax.set_zlabel('AQ')

#plt.savefig('asrs_aq.png', dpi=72, bbox_inches='tight')
plt.show()


## Group feeatures into a dataframe 
import numpy as np
ASRS_X = pd.DataFrame()
ASRS_X ['Inattentive'] = feature_A
ASRS_X ['Hyperactive'] = feature_B
ASRS_X ['AQ'] = feature_C

##################################################################
## K-Means ############################################

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=1).fit(ASRS_X)

label = kmeans.labels_
print(label[:10] )# show first 10 labels 

# ========  Visialization =====
import matplotlib.colors as mcolors
colors = [ 'slateblue', 'forestgreen','goldenrod']
attribute = list(ASRS_X.columns)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(feature_A, feature_B, feature_C, c= label, cmap = mcolors.ListedColormap(colors))
ax.set_xlabel('Inattentive')
ax.set_ylabel('Hyperactive/Impulsive')
ax.set_zlabel('AQ')
 
#plt.savefig('ASRS_data_kmeans.png', dpi=72, bbox_inches='tight')

plt.show()

## K-Means on Winedataset

from sklearn.cluster import KMeans
kmeans_wine = KMeans(n_clusters=3, random_state=1).fit(data_x)

feature_1 = data_x[:,0] # flavanoids
feature_2 = data_x[:,1] # color_intensity
label_wine_kmeans = kmeans_wine.labels_
plt.figure(figsize=(10,8))
classes = ['Class 0 wine', 'Class 1 wine', 'Class 2 wine']
colors = [ 'goldenrod', 'slateblue','forestgreen']
 
plt.scatter(feature_1, feature_2, c= label_wine_kmeans , cmap = mcolors.ListedColormap(colors))

plt.xlabel("flavanoids")
plt.ylabel("color_intensity")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.title('Wine Dataset with Kmeans')
plt.savefig('Wine_data_kmeans.png', dpi=72, bbox_inches='tight')
plt.show()

##################################################################
## Gaussian Mixture Model on Wine Dataset ######################
from sklearn.datasets import load_wine
from matplotlib.colors import LogNorm
from sklearn.mixture import GaussianMixture

winedata = load_wine()
data_x = winedata.data


gmm_clf = GaussianMixture(n_components=3,covariance_type='full')
gmm_clf.fit(data_x)
feature_1 = data_x[:,0] # flavanoids
feature_2 = data_x[:,1] # color_intensity
label = gmm_clf.predict(data_x)

plt.figure(figsize=(10,8))
classes = ['Class 0 wine', 'Class 1 wine', 'Class 2 wine']
colors = [ 'goldenrod', 'slateblue','forestgreen']

x = np.linspace(0., 5.)
y = np.linspace(0., 15.)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -gmm_clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm( ),
                 levels=np.logspace(0,1, 15))
CB = plt.colorbar(CS, shrink=1, extend='both')
plt.scatter(feature_1, feature_2, c= label, cmap = mcolors.ListedColormap(colors))

plt.xlabel("flavanoids")
plt.ylabel("color_intensity")
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.title('Wine Dataset with GMM')
#plt.savefig('Wine_data_GMM.png', dpi=72, bbox_inches='tight')
plt.show()

## Gaussian Mixture Model on ASRS_AQ Dataset ######################
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3,covariance_type='full').fit(ASRS_X)
label2 = gmm.predict(ASRS_X)

from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import axes3d
plt.figure(figsize=(10,8))
 
colors = [ 'forestgreen','goldenrod','slateblue' ]
attribute = list(ASRS_X.columns)
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
 

ax.scatter(feature_A, feature_B, feature_C, c= label2, cmap = mcolors.ListedColormap(colors))
 
ax.set_xlabel('Inattentive')
ax.set_ylabel('Hyperactive/Impulsive')
ax.set_zlabel('AQ')
 
#plt.savefig('ASRS_data_GMM.png', dpi=72, bbox_inches='tight')

plt.show()

##################################################################
### Hierarchical clustering #######################
from sklearn.cluster import AgglomerativeClustering
Hie_clus = AgglomerativeClustering(n_clusters=3, distance_threshold =None, linkage='ward', affinity='euclidean').fit(ASRS_X)

label3 = Hie_clus.labels_

plt.figure(figsize=(10,8))
classes =  [ 'Less likely', 'likely','highly likely']
colors = [ 'forestgreen','slateblue','goldenrod']
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
 

ax.scatter(feature_A, feature_B, feature_C, c= label3, cmap = mcolors.ListedColormap(colors))
 
ax.set_xlabel('Inattentive')
ax.set_ylabel('Hyperactive/Impulsive')
ax.set_zlabel('AQ')
 

#plt.savefig('ASRS_data_Hierarchical.png', dpi=72, bbox_inches='tight')

plt.show()