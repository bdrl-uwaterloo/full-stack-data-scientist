# Multiple Linear Regression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


import os

os.chdir('/Users/rachelzeng/dsbook/AppendixML_Code_Data/Cig_Data')
print (os.getcwd())
#import data-----------------------------------------
Cigdata =pd.read_csv( "cig_data.csv")
print(Cigdata.head())
print(Cigdata.describe())
Cigdata.columns
#Plot 3D-----------------------------------------
fig = plt.figure()
obs = fig.add_subplot(111, projection='3d')

Z = Y = Cigdata.iloc[:, 1].values
X_1 = Cigdata.iloc[:, 0].values
X_2 = Cigdata.iloc[:, 2].values
obs.scatter(X_1, X_2, Z, color = 'r', label = 'Observed data' )
obs.set_xlabel('Father cigarette consumption in the past week')
obs.set_ylabel('# of Parties in the past week ')
obs.set_zlabel('# of Cigarettes in the past week ')
obs.legend()
plt.show()


# 自变量是Cig_data的第一和第三列
X = Cigdata.iloc[:,[0,2]].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# 使用Sklearn在训练集上训练多元线性回归模型
multiLinearregressor = LinearRegression()
multiLinearregressor.fit(X_train, y_train)
print(multiLinearregressor.intercept_, multiLinearregressor.coef_)


alpha = multiLinearregressor.intercept_
beta = multiLinearregressor.coef_[0]
delta = multiLinearregressor.coef_[1]
Yhat = alpha + beta*X_1+ delta*X_2
print(Yhat)

# 创建预测值所在的平面
x1 = np.arange(min(X_1), max(X_1),0.1)
x2 = np.arange(min(X_2), max(X_2),0.1)
X,Y = np.meshgrid(x1,x2)
Zhat = alpha + beta*X+ delta*Y

Z = Cigdata.iloc[:, 1].values
# 预测测试集结果
fig2 = plt.figure(2)
pred = fig2.add_subplot(111, projection='3d')
pred.plot_surface(X, Y, Zhat, color='darkgreen')
pred.scatter(X_1, X_2, Z, color='r')

pred.set_xlabel('Father cigarette consumption per week')
pred.set_ylabel('Number of Parties per week')
pred.set_zlabel('Cigarette consumption per week')
plt.show()

# SGD Regressor----------------------------------------
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
sgd = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-3))

sgd.fit(X_train, y_train)

# Prediction OLS vs SGD Regressor----------------------------------------
from sklearn import metrics
y_pred_ols = multiLinearregressor.predict(X_test)
y_pred_sgd = sgd.predict(X_test)
# Mean squared error
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred_ols)))
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred_ols)))