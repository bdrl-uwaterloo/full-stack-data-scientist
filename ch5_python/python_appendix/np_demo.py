import numpy as np
# Create ndarray
a = np.array([1,2,4,5,6,7])  # from python list
a
type(a)
a = np.array(range(1,12)) # from python list
a
a = np.linspace(0,9,10) # from linspace
a
a = np.arange(0,9,2) # from arange
a
type(a)
# ndarray attributes
one_dim = np.arange(12)
one_dim
one_dim.ndim
one_dim.shape
one_dim.dtype
one_dim.data
# ndarray.reshape
two_dim = one_dim.reshape(3,4)
two_dim
two_dim.ndim
two_dim.shape
two_dim.dtype
two_dim.data
# indexing and slicing
two_dim[1,2] # at the second row and third column
two_dim[1,1:3] # second row from the second to the third column
two_dim[:2,::2] # including step
# Elementwise
x = np.eye(3) # Identity matrix of rank 3
x
y = np.random.rand(3,3) # numbers vary according to seed.
y
x + y # Matrix addition
x * y # elementwise product
x @ y # Matrix product
np.transpose(y) # transpose y
# Useful Methods
z = np.random.rand(3,5)
z
z.sum()
z.min(axis = 0)
z.min(axis = 1)
z.mean()
z.mean(axis = 0)
z.mean(axis = 1)