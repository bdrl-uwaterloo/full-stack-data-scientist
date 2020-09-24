# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_s_curve
colors = ['slateblue', 'goldenrod','forestgreen']
x,y = make_s_curve(2000, random_state=1)
fig = plt.figure(figsize=(6,16))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y, cmap=mcolors.ListedColormap(colors))
ax.view_init(10,-60)


# %%
from sklearn import manifold 
from sklearn.decomposition import KernelPCA
MDS_Y = manifold.MDS(n_components= 3, max_iter=100, n_init = 1).fit_transform(x)
isomap_Y = manifold.Isomap(n_components = 3, n_neighbors = 10).fit_transform(x)
KPCA_Y = KernelPCA(n_components= 3 ).fit_transform(x)


# %%
plt.figure(figsize=(12, 8))
plt.subplot(2,1,1)
plt.title('MDS projection')
plt.scatter(MDS_Y[:,0], MDS_Y[:,1], c=y, cmap=mcolors.ListedColormap(colors))

plt.subplot(2,1,2)
plt.title('Isomap')
plt.scatter(isomap_Y[:,0], isomap_Y[:,1], c=y, cmap=mcolors.ListedColormap(colors))


# %%



