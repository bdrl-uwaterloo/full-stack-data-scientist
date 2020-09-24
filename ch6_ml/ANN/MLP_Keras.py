import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# To check which backend keras is using in keras.json, in our example we are using tensorflow as backend.

# Neural Network can be used to model regression problem. (linear and logistic)
# Back in Linear Regression 
# Recall that we are building layers.
# keras.io/api/models

#from tensorflow.keras.layers import Dense, Flatten # single-input and single output stacks of layers

#Set dir-----------------------------------------
import os
os.chdir('../full-stack-data-scientist/ch6_ml/Data')
print (os.getcwd())

#import data using Pandas-----------------------------------------
Cigdatapd =pd.read_csv( "cig_data.csv")
Cigdatapd.head(2)
Cigdatapd.describe()
Cigdatapd.dtypes
col_name = list( Cigdatapd.columns )

#import data using tf.data.Dataset-----------------------------------------
Train_set = Cigdatapd.sample(frac = 0.8, random_state = 0)
Test_set = Cigdatapd.drop (Train_set.index)

# Normalize----------------------------------------
def Mean_Std (col_name, dataset):
   return [dataset[col_name].mean(), dataset[col_name].std()]
Train_mean = Train_set[col_name].mean()
Train_std = Train_set[col_name].std()
def normalize (col_name,dataset):
   return (dataset - Mean_Std(col_name,dataset)[0])/ Mean_Std(col_name,dataset)[1]
Train_normset = normalize (col_name, Train_set)
Test_normset = normalize (col_name , Test_set)
Train_target = Train_normset.pop('Cigaratte consumption per week')
Test_target = Test_normset.pop('Cigaratte consumption per week')

#############################################################################
#--- Linear Regression Replication using Keras ------
# A sequential model is appropriate to use in this case, it is a simple stake of layers in which each layers has a single input and output unit.

# Method 1:
lin_reg1 = tf.keras.Sequential( )

# Output layer
lin_reg1.add(layers.Dense(1,  activation = 'relu', input_dim =2, use_bias =True))

#---------------------------------------------

# Method 2: equivalently, we can assemble that layer into the model
lin_reg = tf.keras.Sequential([
  #输出
  layers.Dense(1, activation = 'linear', input_dim =2, use_bias =True)
])
# input_shape = (2, ) 等于 input_dim = 2
#--------------------------------------------
# 用损失和随机梯度下降优化器编译模型
LearnRate= 0.01 
tf.keras.optimizers.SGD(
    learning_rate=LearnRate, momentum=0.0, nesterov=False, name='SGD')

lin_reg.compile(loss= 'mean_squared_error', # 成本函数 
   optimizer ='SGD', #  或直接使用默认参数, 'sgd'.
   metrics=['mae'] #  mae =平均绝对误差
   )
# summary
lin_reg.summary()

# Training
#Epochs- forward backward passes
history = lin_reg.fit(Train_normset, Train_target, epochs=500, batch_size=900,  verbose=1, validation_split=0.25)

W, b = lin_reg.layers[0].get_weights()
print(W)

import matplotlib.pyplot as plt
print(history.history.keys())
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
      
#####Preidiction############################################################

Pred_test = lin_reg.predict(Test_normset).flatten()

#Plot 3D-----------------------------------------
fig = plt.figure(1)
obs = fig.add_subplot(111, projection='3d')

Z = np.array(Test_target )
X_1 = Test_normset.iloc[:, 0].values
X_2 = Test_normset.iloc[:, 1].values
obs.scatter(X_1, X_2, Z, color = 'r', label = ' Test Observed data' )
obs.scatter(X_1, X_2, Pred_test,color='darkgreen')
obs.set_xlabel('Father cigarette consumption in the past week')
obs.set_ylabel('# of Parties in the past week ')
obs.set_zlabel('# of Cigarettes in the past week ')
obs.legend()
plt.show()


#Compare with Normal 
from sklearn.linear_model import LinearRegression
multiLinearregressor = LinearRegression()
multiLinearregressor.fit(Train_normset, Train_target)
coef = multiLinearregressor.coef_

#predict 
Mlinreg_preidct = multiLinearregressor.predict(Test_normset)
#Plot 3D-----------------------------------------
fig = plt.figure(2)
obs = fig.add_subplot(111, projection='3d')

Z = np.array(Test_target )
X_1 = Test_normset.iloc[:, 0].values
X_2 = Test_normset.iloc[:, 1].values
obs.scatter(X_1, X_2, Z, color = 'r', label = ' Test Observed data' )
obs.scatter(X_1, X_2, Mlinreg_preidct ,color='blue')
obs.set_xlabel('Father cigarette consumption in the past week')
obs.set_ylabel('# of Parties in the past week ')
obs.set_zlabel('# of Cigarettes in the past week ')
obs.legend()
plt.show()

# Prediction Error 
Err = Mlinreg_preidct - Test_target
Err_NN = Pred_test - Test_target
fig = plt.figure (3)
plt.hist(Err, bins = 30, alpha = 0.3, label = "Multi-linear error")
plt.hist(Err_NN, bins  = 30, alpha = 0.3, label = 'ANN-linear error')
plt.xlabel ("Prediction Error")
plt.legend(loc = 'upper right')
plt.show()
print ("Weights for Single-layer Nerual Network:{}".format(W))
print ("Total Error for Single-layer Nerual Network:{}".format(sum(abs(Err_NN))))
print ("Coefficients for multilinear regression:{}".format(coef))
print ("Total Error for multilinear regression:{}".format(sum(abs(Err))))

#############################################################################
# creating model using Functional API
inputs = keras.Input(shape=(2,))
Output = layers.Dense(1,activation='linear')(inputs)

lin_reg2 = keras.Model(inputs=inputs,outputs=Output)
#sgd=keras.optimizers.SGD()
lin_reg2.compile(optimizer='sgd' ,loss='mse',metrics=['mse'])
lin_reg2.summary()

lin_reg2.fit(Train_normset, Train_target, epochs=300, batch_size=800,  verbose=1, validation_split=0.25)


