#Set dir-----------------------------------------
import os
os.chdir('C:\\Users\\rache\\Downloads\\dsbook\\AppendixML_Code_Data\\Cig_Data')
# Usking sklearn
#import data using Pandas----------------------------------------
import pandas as pd
Cigdatapd =pd.read_csv( "cig_data.csv")
col_name = list(Cigdatapd.columns)
#Split Data--------------------------------------
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit_transform(Cigdatapd.values)
Cigdatapd = pd.DataFrame (scaler )
Cigdatapd.columns = col_name

Train_set = Cigdatapd.sample(frac = 0.8, random_state = 0)
Test_set = Cigdatapd.drop (Train_set.index)
Train_target = Train_set.pop('Cigaratte consumption per week')
Test_target = Test_set.pop('Cigaratte consumption per week')

from sklearn.linear_model import SGDRegressor
epoch_num = 100000
learning_rate = 0.01
sgd_lin_reg = SGDRegressor (max_iter=epoch_num , tol = 0.001,eta0 =learning_rate)
sgd_lin_reg.fit(Train_set,Train_target)
print( [sgd_lin_reg.intercept_, sgd_lin_reg.coef_])


##Tensorflow #########
import numpy as np
import tensorflow.compat.v1 as tf # if you are using TensorFlow 2x
tf.disable_v2_behavior() 
import pandas as pd
import matplotlib.pyplot as plt

#As before we have our dataset transform into a matrix
Cigdata = np.matrix(pd.read_csv( "cig_data.csv").values)
X_data = Cigdata[:, [0,2]].transpose()
y_data = Cigdata [:, 1].transpose()

X_1 = Cigdata[:, 0]
X_2 = Cigdata[:, 2]
x1 = np.arange(min(X_1), max(X_1),0.1)
x2 = np.arange(min(X_2), max(X_2),0.1)
X,Y = np.meshgrid(x1,x2)

col_num = 2
samples_size = Cigdata.shape [0]

def linear_regression():
    # We create a placeholders so that we can assign values later
  x = tf.placeholder(tf.float32, shape=(col_num, None ), name='x') # 2 independent variables
  y = tf.placeholder(tf.float32, shape=(1, None), name='y')
  with tf.variable_scope('lreg') as scope:
    # Randomly draw a value from the normal distirbution, notice that the Variable() constructor will need an initial value that can be a Tensor of any type and shape.
    w = tf.get_variable("w", shape=(1, col_num),initializer=tf.random_normal_initializer())
    b = tf.get_variable("b", shape=(),initializer=tf.random_normal_initializer())

    y_pred = tf.add(tf.matmul(w, x), b)  # or tf.matmul(w, x) + b
    loss = tf.reduce_mean(tf.square(y_pred - y))
  return x, y, y_pred, loss,w,b

#Now we will define the hyperparameters of the model, the Learning Rate, the number of Epochs, and the baatch size.
num_epochs = 300000
batch_size = 500
learning_rate = 0.00001

def run():
  x, y, y_pred, loss,w,b = linear_regression()

  opt_min = tf.train.GradientDescentOptimizer (learning_rate).minimize(loss)
  #Allows to execute graphs in which tensors are processed through operations.
  with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer()) 
    # Only if this run, our variables will hold the values we previously decleared
    feed_dict = {x: X_data, y: y_data} 
    # it is used to override the placeholder value of x and y.
		
    for epoch in range(num_epochs):
       _, current_loss, current_w, current_b = sess.run([opt_min, loss ,w,b ],feed_dict) # ready to run epoch by calling session.run()
       print('At', epoch, 'epoch,', "the loss is ",current_loss,  ', w =',current_w, ', and b=',  current_b)
if __name__ == "__main__":
  run()


