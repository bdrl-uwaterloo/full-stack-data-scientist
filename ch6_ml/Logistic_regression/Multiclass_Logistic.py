###Tensorflow Gradient #################
import tensorflow as tf
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np


winedata = load_wine()


X_data =  winedata["data"]
y_data = winedata["target"]
classes = list(winedata.target_names)
feature_names = winedata.feature_names
print('Classes: {}'.format(classes), 'Feature names: {}'.format(feature_names))

########Plot##############################################################

def LASSO_index(X, y):
    Lasso_algo = LassoCV().fit(X, y)
    shrinking = np.abs(Lasso_algo.coef_)
    idx_third = shrinking.argsort()[-3] # we want the coefficients above third highest coefficients
    features_index = (-shrinking).argsort()[:2]
    return (features_index)



########Plot##############################################################
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

fea_index = LASSO_index(X_data, y_data)
X = X_data[:, fea_index]
feature_1 =  X[:,0] # 类黄酮
feature_2 = X[:,1] # 颜色强度
label = y_data


# 模型训练
multiclass_logreg = LogisticRegression(multi_class = 'multinomial', solver = 'sag')
multiclass_logreg.fit(X, label)
#### 

# 绘制决策边界。为此，我们将为每种颜色分配一种颜色
# 制作网格[x_min，x_max] [y_min，y_max]。
x_min, x_max = feature_1.min() - .5, feature_1.max() + .5
y_min, y_max = feature_2.min() - .5, feature_2.max() + .5
h = .01  # step size in the mesh
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = multiclass_logreg.predict(np.c_[x_grid.ravel(), y_grid.ravel()])

# Put the result into a color plot
Z = Z.reshape(x_grid.shape) 
plt.figure(2)
backcolors= ['palegreen', 'azure', 'lemonchiffon']
plt.pcolormesh(x_grid, y_grid, Z, cmap= mcolors.ListedColormap(backcolors))

classes = ['Class 0 wine', 'Class 1 wine', 'Class 2 wine']
colors = [ 'forestgreen','slateblue', 'goldenrod']
scatter = plt.scatter(feature_1, feature_2, c= label, cmap = mcolors.ListedColormap(colors))
plt.xlabel(feature_names[fea_index[0]])
plt.ylabel(feature_names[fea_index[1]])
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.show()

##Prediction
multiclass_logreg.predict_proba([[0,8]])
multiclass_logreg.predict_proba([[2,2]])
multiclass_logreg.predict_proba([[3,6]])


