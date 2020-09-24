import numpy as np
from matplotlib import pyplot as plt
 
def sigmoid(x):
    return 1/(1+np.exp(-x))
 
x=np.linspace(-10,10,100)
y=sigmoid(x)
plt.plot(x,y,'b')
plt.title('Sigmoid Function plot')
plt.show()

# Binary Classifers 
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
winedata = load_wine()

X_data = winedata["data"]
y_data = winedata["target"]
list(winedata.target_names)
feature_names = winedata.feature_names
print(feature_names)
print(y_data)

# 现在让我们看一个功能，只关注 class 0葡萄酒
# 请注意，此数据集所有分类变量均已编码为数字，
# 如果没有，请参考OneHotEncoder（）教程
nth_feature = 6 # 选择第六个特征变量
x1 = X_data [:,nth_feature]
y1 =(y_data == 0).astype(np.int)

# Plot
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 现在让我们在只有一个特征/预测变量/自变量的情况下进行决策from sklearn.linear_model import LogisticRegression
Logistic_reg = LogisticRegression()
 # 强制将预测变量转换成1列，-1表示行数未知。
x1 = x1.reshape(-1,1)
Logistic_reg.fit(x1, y1)

x1_newdata = np.linspace(x1.min(),x1.max(),1000).reshape(-1,1)
y1hat = Logistic_reg.predict_proba(x1_newdata)

classes = ['Not Class 0 wine', 'Class 0 wine']
colors = [ 'green','blue']
fig= plt.figure(figsize=(6,3))
obs = plt.scatter(x1, y1,  c=y1, cmap = mcolors.ListedColormap(colors))
plt.plot(x1_newdata, y1hat[:,1],'b-', label = 'Class 0 wine')
plt.plot(x1_newdata, y1hat[:,0],'g--', label = 'Not Class 0 wine')
plt.legend(handles=obs.legend_elements()[0], labels=classes, numpoints=1, fontsize=8)
plt.xlabel(feature_names[nth_feature])
plt.ylabel('Probability')
plt.show()


##############################################################################
# Find importance of the features using LASSO 
# ---------------------------------------------------------
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html
# To decide on the importance of the features we are going to use LassoCV
# estimator. The features with the highest absolute `coef_` value are
# considered the most important
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.feature_selection import SelectFromModel


##############################################################################
# Select from the model features with the higest score
# ---------------------------------------------------------
#
# Now we want to select the two features which are the most important.
# SelectFromModel() allows for setting the threshold. Only the features with
# the `coef_` higher than the threshold will remain. Here, we want to set the
# threshold slightly above the third highest `coef_` calculated by LassoCV()
# from our data.

Lasso_algo = LassoCV().fit(X_data, y_data)
shrinking = np.abs(Lasso_algo.coef_)
# LASSO方法取头个两个重要的变量
idx_third = shrinking.argsort()[-3] 
features_index = (-shrinking).argsort()[:2]
features_name = np.array(feature_names)[features_index]
print('Selected features: {}'.format(features_name ))

X = X_data[:, features_index]
feature_1 =  X[:,0]
feature_2 = X[:,1]
label = y1

plt.figure(1)
classes = ['Not Class 0 wine', 'Class 0 wine']
colors = [ 'green','blue']
scatter = plt.scatter(feature_1, feature_2, c= label, cmap = mcolors.ListedColormap(colors))
plt.xlabel(feature_names[features_index[0]])
plt.ylabel(feature_names[features_index[1]])

plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.show()


Logistic_reg2 = LogisticRegression()
Logistic_reg2.fit(X,y1)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = feature_1.min() - .5, feature_1.max() + .5
y_min, y_max = feature_2.min() - .5, feature_2.max() + .5
h = .01  # step size in the mesh
x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = Logistic_reg2.predict(np.c_[x_grid.ravel(), y_grid.ravel()])

# Put the result into a color plot
Z = Z.reshape(x_grid.shape) 
plt.figure(2)
backcolors= ['palegreen', 'azure']
plt.pcolormesh(x_grid, y_grid, Z, cmap= mcolors.ListedColormap(backcolors))

# Plot also the training points
scatter = plt.scatter(feature_1, feature_2, c= label, cmap = mcolors.ListedColormap(colors))
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.xlabel(feature_names[features_index[0]])
plt.ylabel(feature_names[features_index[1]])

plt.xlim(x_grid.min(), x_grid.max())
plt.ylim(y_grid.min(), y_grid.max())

plt.show()
