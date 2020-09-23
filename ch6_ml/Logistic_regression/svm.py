# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# %%
winedata = load_wine()
X = winedata['data']
X = X[:, [6,9]]
y = winedata.target
classes = list(winedata.target_names)


# %%
x_train, x_test, y_train, y_test = train_test_split(X,y, random_state =1)


# %%
svmclf = svm.SVC(kernel = 'poly', C =1).fit(x_train, y_train)
softmax = LogisticRegression(multi_class = 'multinomial', solver = 'sag').fit(x_train,y_train)


# %%
svm_pred = svmclf.predict(x_test)
softmax_pred = softmax.predict(x_test)


# %%
print (accuracy_score (y_test ,svm_pred))


# %%
print (accuracy_score (y_test ,softmax_pred))


# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

feature_1 =  X[:,0] # flavanoids
feature_2 = X[:,1] # color_intensity
x_min, x_max = feature_1.min() - .5, feature_1.max() + .5
y_min, y_max = feature_2.min() - .5, feature_2.max() + .5
h = .01  # step size in the mesh
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svmclf.predict(np.c_[x_grid.ravel(), y_grid.ravel()])

# Put the result into a color plot
Z = Z.reshape(x_grid.shape) 
plt.figure(2)
backcolors= ['palegreen', 'azure', 'lemonchiffon']
plt.pcolormesh(x_grid, y_grid, Z, cmap= mcolors.ListedColormap(backcolors))

classes = ['Class 0 wine', 'Class 1 wine', 'Class 2 wine']
colors = [ 'forestgreen','slateblue', 'goldenrod']
scatter = plt.scatter(feature_1, feature_2, c= y, cmap = mcolors.ListedColormap(colors))
plt.xlabel('flavanoids')
plt.ylabel('color_intensity')
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.show()

