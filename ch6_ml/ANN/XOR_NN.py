#############################################################################
#--- XOR Classification problem using Keras ------
#XOR problem dataset
import numpy as np
np.random.seed(0)
n=100
x1 = np.random.rand(n,2) * (-1)
x2 = np.random.rand(n,2)
x2[:,1] *= (-1)
x3 = np.random.rand(n,2)
x3[:,0] *= (-1)
x4 = np.random.rand(n,2)
x = np.concatenate((x1, x2, x3, x4))

y1 = np.ones(n)
y4 = np.ones(n)
y2 = np.zeros(n)
y3 = np.zeros(n)
y = np.concatenate((y1,y2,y3,y4))
print (x1[[1,2],:])
print (x2[[1,2],:])
print (x3[[1,2],:])
print (x4[[1,2],:])
#Plot  ------ ------ ------ ------
import matplotlib.pyplot as plt
plt.scatter(x1[:,0], x1[:,1], color ='turquoise', marker = 'o')
plt.scatter(x2[:,0], x2[:,1], color ='salmon', marker = 'o')
plt.scatter(x3[:,0], x3[:,1], color ='salmon', marker = 'o')
plt.scatter(x4[:,0], x4[:,1], color ='turquoise', marker = 'o')
plt.show()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

XOR_Clf = tf.keras.Sequential()
XOR_Clf.add(layers.Dense(64, activation = 'relu', input_dim =2, use_bias =True))
XOR_Clf.add(layers.Dense (32,activation = 'sigmoid'))
XOR_Clf.add(layers.Dense (1, activation = 'linear', name = 'Output'))

# Equivalently: 
XOR_Clf = tf.keras.Sequential([
  layers.Dense(64, activation = 'relu', input_dim =2, use_bias =True),
  layers.Dense (32,activation = 'sigmoid' ),
  layers.Dense (1, activation = 'linear', name = 'Output')
])
XOR_Clf.summary()

XOR_Clf.compile(optimizer='sgd', loss='mse',metrics=['accuracy'])
history = XOR_Clf.fit(x, y, batch_size = 1, epochs = 100)

XOR_Clf.layers[0].get_weights()
XOR_Clf.layers[1].get_weights()

print(XOR_Clf.predict_proba(x))
#############################################################################
import matplotlib.colors as mcolors

def plot_decision_boundary(pred_func):
# Set min and max values and give it some padding
   x_min, x_max = x[:, 0].min() - .5, x [:, 0].max() + .5
   y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
   h = 0.01
   # Generate a grid of points with distance h between them
   x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
   # Predict the function value for the whole gid
   Z = pred_func(np.c_[x_grid.ravel(),  y_grid.ravel()])
   Z = Z.reshape(x_grid.shape)
   # Plot the contour and training examples
   backcolors= ['peachpuff', 'azure']
   colors = ['salmon', 'turquoise']
   plt.contourf(x_grid, y_grid, Z, cmap=mcolors.ListedColormap(backcolors))
   plt.scatter(x[:, 0], x[:, 1], c=y, cmap= mcolors.ListedColormap(colors))


plot_decision_boundary(lambda x: XOR_Clf.predict_classes(x))
plt.show()



#Functional API
# 定义输入元素
Input_1 = tf.keras.Input(shape=(1,))
Input_2 = tf.keras.Input(shape=(1,))
# 隐藏层1连接隐藏层2， 将输入2连接隐藏层1。
Hidden_1 = layers.Dense (64, activation = 'relu')(Input_2) 
Hidden_2 = layers.Dense (32, activation = 'sigmoid')(Hidden_1)
# 通过串联合并所有功能
comb = tf.keras.layers.concatenate([Input_1, Hidden_2])
# 全联接输出
Output = tf.keras.layers.Dense(1, activation = 'linear', name = 'ouput')(comb)
# 制定具有两个输入功能和一个输出的Functional模型。
Fun_XOR_Clf = tf.keras.Model(inputs= [Input_1,Input_2], outputs = [Output])

Fun_XOR_Clf.summary()
