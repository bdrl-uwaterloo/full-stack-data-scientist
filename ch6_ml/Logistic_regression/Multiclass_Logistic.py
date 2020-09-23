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

fea_index = LASSO_index(X_data, y_data)

########Plot##############################################################
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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


##############################################################################


features_name = np.array(feature_names)[fea_index]
X = np.matrix(X_data[:,fea_index])
feature_1 =  X[:,0]
feature_2 = X[:,1]
X_LASSO = X.transpose() #Select our features using LASSO
y_data =np.matrix(y_data)
num_classes = len(classes)
num_feature = 2


##############################################################################
# Training parameters
num_epochs = 300000
batch_size = 70
learning_rate = 0.001

# Initial values for weights and bias

def logistic_regression():
    # We create a placeholders so that we can assign values later
    x = tf.placeholder(tf.float32, shape=(num_feature, None ), name='x') # 2 independent variables
    y = tf.placeholder(tf.float32, shape=(1, None), name='y')
    with tf.variable_scope('logreg') as scope:
    #从正态分布中随机抽取一个值，请注意，Variable（）构造函数将需要一个初始值，该初始值可以是任何类型和形状的Tensor。        w = tf.get_variable("w", shape=(1, num_feature),initializer=tf.random_normal_initializer())
        b = tf.get_variable("b", shape=(),initializer=tf.random_normal_initializer())

        y_pred = tf.nn.softmax(tf.add(tf.matmul(w, x), b)) # 应用softmax回归
        loss = tf.reduce_mean(-tf.reduce_sum(y * tf.math.log(y_pred)))
    return x, y, y_pred, loss,w,b


def run():
  x, y, y_pred, loss,w,b = logistic_regression()

  opt_min = tf.train.GradientDescentOptimizer (learning_rate).minimize(loss)
  #Allows to execute graphs in which tensors are processed through operations.
  with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer()) 
    # Only if this run, our variables will hold the values we previously decleared
    feed_dict = {x: X_LASSO, y: y_data} 
    # it is used to override the placeholder value of x and y.
		
    for epoch in range(num_epochs):
       _, current_loss, current_w, current_b = sess.run([opt_min, loss ,w,b ],feed_dict) # ready to run epoch by calling session.run()
       print('At', epoch, 'epoch,', "the loss is ",current_loss,  ', w =',current_w, ', and b=',  current_b)
if __name__ == "__main__":
  run()

