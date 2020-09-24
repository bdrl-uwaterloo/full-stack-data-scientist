# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# load Wine Dataset
from sklearn.datasets import load_wine
from sklearn.neighbors import  KNeighborsClassifier
import numpy as np
winedata = load_wine()
feature_names = winedata.feature_names


# %%
# derive class 1 wine data
inx = np.where(data_y ==1 )[0]
class_1_y = data_y [inx ]
class_1_x = data_x [inx,]


# %%
from sklearn.ensemble import IsolationForest

clf = IsolationForest( contamination = 'auto')
clf.fit(class_1_x)
IFprediction = clf.predict(class_1_x)
anom_ind = np.where(IFprediction < 0)

anom_ind


# %%

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.figure(figsize=(10,8))
 
scatter= plt.scatter(class_1_x[:,0], class_1_x[:,1],  c='slateblue', marker='o', s=150. ,label='class 1 wine')
scatter2 = plt.scatter(class_1_x[anom_ind,0], class_1_x[anom_ind,1], c='r',  marker='o', s=150.,label='Abnormal')
plt.xlabel("flavanoids")
plt.ylabel("color_intensity")
plt.legend()
plt.savefig('wine_data_iForest.png', dpi=72, bbox_inches='tight')
plt.show()


# %%



