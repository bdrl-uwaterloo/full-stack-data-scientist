import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from sklearn.metrics import accuracy_score

df = pd.read_csv('datasets_heart.csv')

sns.countplot(x='sex',data=df,palette="muted")
plt.xlabel(" Woma / Man")

sns.countplot(hue='sex',x='target', data=df,palette="muted")
plt.legend(labels=['Female', 'Male'])

female_data= df [ df['sex']==0] #Only consider female
female_data.info()

feature_lis = ['trestbps','oldpeak','age']
feature_1 = female_data[feature_lis[0]]
feature_2 =  female_data[feature_lis[1]]
feature_3 =    female_data[feature_lis[2]]
target =   female_data['target']



from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X, target)
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

from sklearn.ensemble import IsolationForest

isofo_clf = IsolationForest( contamination = 0.08)

isofo_clf.fit(X)
y_pred = isofo_clf.predict(X)
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
