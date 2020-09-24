from sklearn.datasets import load_wine
from sklearn.neighbors import  KNeighborsClassifier
import numpy as np
winedata = load_wine()
feature_names = winedata.feature_names
print(feature_names)

from sklearn.linear_model import LassoCV, LogisticRegression
data_x = winedata.data
data_y = winedata.target
Lasso_algo = LassoCV().fit(data_x, data_y)
shrinking = np.abs(Lasso_algo.coef_)
features_index = (-shrinking).argsort()[:2]

data_x = data_x [:,features_index ]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=10, weights= "uniform")
knn.fit(x_train,y_train)

# Compare with Logistic Regression 
from sklearn.metrics import classification_report
y_pred = knn.predict(x_test)
print(classification_report(y_test, y_pred, target_names=winedata.target_names))

from sklearn.linear_model import LogisticRegression
Logistic_reg = LogisticRegression()
Logistic_reg.fit(x_train,y_train)
y_pred2 = Logistic_reg.predict(x_test)
print(classification_report(y_test, y_pred2, target_names=winedata.target_names))