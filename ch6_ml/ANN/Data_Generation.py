# Create Cig_data for regression --------------
import numpy as np
import pandas as pd




# Gnerate regression dataset--------------
sample = 1500
father_com = np.random.randint (low=0, high = 200, size= sample )
parties = np.random.randint (low= 0, high= 10, size = sample )
Cig_com = 1.1* father_com + 4.1*parties +  np.random.normal(50,100,sample)
Cig_com = Cig_com.astype(int)
data_linreg=  np.array([ father_com,Cig_com,parties]).T
Cig_Data = pd.DataFrame(data_linreg, columns = [ 'Father cigaratte consumption per week', 'Cigaratte consumption per week','Number of Parties attended per week'])
Cig_Data[Cig_Data < 0] = 0
Cig_Data.head()
import os
os.chdir('C:\\Users\\rache\\Downloads\\dsbook\\AppendixML_Code_Data\\Cig_Data')
print (os.getcwd())
Cig_Data.to_csv('cig_data.csv',index=False)
# Create Cig_data for regression --------------

from sklearn.datasets.samples_generator import make_regression
import matplotlib.pyplot as plt 
X, y = make_regression (n_samples=1000, n_features =2, n_targets= 1 )


# Gnerate classification dataset--------------
from sklearn.datasets.samples_generator import make_classification
import matplotlib.pyplot as plt 
X, y = make_classification(n_samples = 1000, n_redundant =0,n_features = 2, n_classes = 3, n_clusters_per_class = 1)
plt.scatter(X[:,0 ], X[:,1], marker = 'o', c=y)
plt.show()

# Gnerate non-linearly seperable classification dataset--------------
from sklearn.datasets.samples_generator import make_gaussian_quantiles

X, y = make_gaussian_quantiles(n_samples = 1000,n_features = 2, n_classes = 3, mean =[10,5], cov= 2)
plt.scatter(X[:,0 ], X[:,1], marker = 'o', c=y)
plt.show()


# XOR problem dataset
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
import matplotlib.pyplot as plt
plt.scatter(x1[:,0], x1[:,1], color ='turquoise', marker = 'o')
plt.scatter(x2[:,0], x2[:,1], color ='salmon', marker = 'o')
plt.scatter(x3[:,0], x3[:,1], color ='salmon', marker = 'o')
plt.scatter(x4[:,0], x4[:,1], color ='turquoise', marker = 'o')
plt.show()