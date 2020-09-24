import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 20
sigma = 5
x_min = -25
x_max = 25
z_min = (x_min + mu ) / sigma
z_max = ( x_max - mu ) / sigma
z_min, z_max

x = np.arange(z_min, z_max, 0.001) # 1 standard deviation away from the mean
x_all = np.arange(-10, 10, 0.001)  # entire range of x, both in and out of spec
y = norm.pdf(x,0,1)
y2 = norm.pdf(x_all,0,1)

##########################################
###### Normal Gaussian Curve ######
fig, ax = plt.subplots(figsize=(9,6))
plt.style.use('fivethirtyeight')
ax.plot(x_all,y2)

ax.fill_between(x,y,0, alpha=0.3, color='b')
ax.fill_between(x_all,y2,0, alpha=0.1)
ax.set_xlim([-4,4])
ax.set_xlabel('# of Standard Deviations')
ax.set_yticklabels([])
ax.set_title('Normal Gaussian Curve')

plt.savefig('normal_curve.png', dpi=72, bbox_inches='tight')
plt.show()

##########################################
## Normal Gaussian Curve with different mean and variance ######
y = norm.pdf(x_all,-1,1)
y2 = norm.pdf(x_all,1,1)
y3 = norm.pdf(x_all,0,2)

fig, ax = plt.subplots(figsize=(9,6))
plt.style.use('fivethirtyeight')
ax.plot(x_all,y, color = 'r', label = "mu = -1, sd = 1")
ax.plot(x_all,y2,color='b', label = "mu = 1, sd = 1")
ax.plot(x_all,y3,color='g', label = "mu = 0, sd = 2")

ax.set_xlim([-6,6])
ax.set_xlabel('# of Standard Deviations')
ax.set_yticklabels([])
ax.set_title('Normal Curve')
plt.legend()

plt.savefig('normal_curve_diff.png', dpi=72, bbox_inches='tight')
plt.show()


##########################################
######## Multivariate Gaussian ######
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
mu = np.array([0, 0])
cov = np.array([[1, .5],[.5, 1]])
Z = multivariate_normal.pdf(pos, mu, cov)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(X, Y, Z, cmap='rainbow', rstride=1, cstride=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.savefig('multigaussian.png', dpi=72, bbox_inches='tight')
fig.show()

fig = plt.figure(figsize=(10,10))
con = fig.add_subplot(111)
con.contour(X, Y, Z,cmap='magma')
fig.savefig('multigaussian_contour.png', dpi=72, bbox_inches='tight')

plt.show()


fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3,  cmap='rainbow',cstride=3, linewidth=1, antialiased=True)
ax.contourf(X, Y, Z, zdir='z', offset=-0.06)
ax.set_zlim(-0.08,0.18)
ax.view_init(20, -21)

fig.savefig('Gauss_contour.png', dpi=72, bbox_inches='tight')
fig.show()