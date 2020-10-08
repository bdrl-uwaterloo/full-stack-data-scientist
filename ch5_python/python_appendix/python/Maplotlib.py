# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Data Visualizations with Matplotlib
# ## Installation
# ### python -m pip install -U pip
# ### python -m pip install -U matplotlib

# %%
### Data visualization has recieved an enormous amount of attention in business sectors and many research fields, as it combines data processing and visual representation which can effectively capture and provides insights on structure, relationship, pattern, anomaly detection and many more applications in datasets. This section is a basic tutorial of Matplotlib, you will learn more plotting techniques as you read along! Here, we will show you how to create 1D, 2D and 3D plots, multi-subplots, configuretaion properties, change of background colors, coun  Without further ado, let's get started.


# %%
import matplotlib.pyplot as plt
import numpy as np
import os 
os.chdir('/Users/rachelzeng/dsbook/fig')


# %%
# 1D horozontal plot
plt.hlines(1,1,20)  # Draw a horizontal line
knots = [1,3,9,15,17,18,20]
plt.xlim(0,21)
plt.ylim(0.5,1.5)

y = np.ones(np.shape(knots))   # Make all y values the same
plt.plot(knots,y,marker ='o', color = 'black')  
ax = plt.gca()
ax.axes.yaxis.set_visible(False)
plt.savefig('1D_plot.png')
plt.show()

# %% [markdown]
# ### The marker is to show how do you wants the dots to be presented, you can try out '^', '|'. 
# ### plt.show() is simply show the plot, plt.savefig will save this plot with file name of '1D_plot.png' in your working directory.

# %% [markdown]
# ### Let draw a simple line with a title and axis lables.

# %%
# A simple linear Line
y = np.linspace(0, 10, 20)  # 20 points with y-axis range from 0 to 10
plt.plot(y)
plt.title('This is a Title' , fontsize = 14)     
plt.xlabel('This is index', fontsize = 12)            
plt.ylabel('This is Y-values' , fontsize = 12)
plt.savefig('aline.png') 


# %%
### Let draw scatter 
### Let draw scatter 
x = np.linspace(0, 5, 20) # 20 points with X-axis range from 0 to 5
i =0
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    plt.plot(x[i],y[i], marker,
             label="marker='{0}'".format(marker))
    i = i+1
plt.legend(loc='best',fontsize=9)
plt.title('This is a scatterplot' , fontsize = 14)     
plt. xlabel('This is X-values', fontsize = 12)            
plt.ylabel('This is Y-values' , fontsize = 12) 
plt.savefig('ascatter.png') 
# %%
## Let draw put dots on the line. 

plt.plot(x, y)
plt.plot(x, y, 'o') 
plt.title('This is a figure of dots on line' , fontsize = 14)     
plt. xlabel('This is X-values', fontsize = 12)            
plt.ylabel('This is Y-values' , fontsize = 12) 
plt.savefig('ascatterline.png') 
plt.show()


# %%
y= np.linspace(0, 10, 20)  # 20 points with y-axis range from 0 to 10
y2 = 1/2 * y +1 
plt.plot(y, linestyle= '-.',label = 'Line 1')
plt.plot(y+1,':', linewidth = 10, alpha=0.8, label = 'Line 2')
plt.plot(y+2,'-', linewidth = 5, alpha=0.5,label = 'Line 3')
plt.plot(y+3,'--', label = 'Line 4')
#Lienwidth controls for the width of aline.
plt.title('Two lines' , fontsize = 14)     
plt. xlabel('This is index', fontsize = 12)            
plt.ylabel('This is Y-values' , fontsize = 12)
plt.legend(loc = 'best')
# loc is the location of legend: 
# 'best'         : 0, (only implemented for axes legends)
# 'upper right'  : 1,
# 'upper left'   : 2,
# 'lower left'   : 3,
# 'lower right'  : 4,
# 'right'        : 5,
# 'center left'  : 6,
# 'center right' : 7,
# 'lower center' : 8,
# 'upper center' : 9,
# 'center'       : 10,
plt.savefig('2lines.png') 

# %%
# Incert a lable
y3 = [i*2-5 for i in y]
plt.plot(y, linestyle= '-.',label = 'Line 1')
plt.plot(y3, label = 'Line 2')
plt.text (3,7, 'Two lines intersect here!', fontsize =10, color ='green')
plt.arrow(4,6.5, 4,-1,head_width = 0.2,ec ='green' )
plt.text (10,-3, '(10,-3)', fontsize =10, color ='black')plt.title('Two lines' , fontsize = 14)     
plt.arrow(10,-4, 8,1,head_width = 0.2,ec ='black' )
plt. xlabel('This is index', fontsize = 12)            
plt.ylabel('This is Y-values' , fontsize = 12)
plt.legend(loc = 'best')

plt.savefig('plt_text.png') 


# %%
# Gradient color
plt.scatter(y, y3, c=y3, cmap=plt.cm.Blues, edgecolor='none' , s=50)


# %%
## Plot 2D data on 3D


# %%
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize= (10,10))
ax = fig.gca(projection='3d')
ax.scatter(y, y3, zs=0 ,label='z=0')
ax.scatter(y, y3, zs=0.2 ,label='z=0.2')
ax.scatter(y, y3, zs=0.4 ,label='z=0.4')
ax.set_zlim(0, 0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
# Customize the view angle so it's easier to see that the scatter points lie
# on the plane y=0
ax.view_init(elev=20., azim=-35)
plt.savefig('3D_scatter.png') 

# %% [markdown]
# ### For more example please visit \url{https://matplotlib.org/gallery/index.html#mplot3d-examples-index}
# %% [markdown]
# ## Multi-subgraphs

# %%
fig, (plt1, plt2) = plt.subplots(2,1) 
plt1.plot(y)
plt2.plot(y3)
plt1.set_title('First plot')
plt2.set_title('second plot')
plt.savefig('2by1plot.png') 


# %%
fig, (plt1, plt2) = plt.subplots(1,2) 
plt1.plot(y)
plt2.plot(y3)
plt1.set_title('First plot')
plt2.set_title('second plot')
plt.savefig('2by1plot.png') 


# %%
fig2, ((plt1, plt2),(plt3, plt4)) = plt.subplots(2,2) 
plt1.plot(y3,y, linestyle='-.',linewidth=3,color = 'sienna')
plt2.plot(y,y3, linewidth=5,color = 'peachpuff')
plt4.scatter(y,y3, marker='^',color = 'gold')
plt3.scatter(y3,y, marker = 'p',color = 'palevioletred')
plt.savefig('4by4plot.png') 

# %% [markdown]
# ## Meshgrid

# %%
import matplotlib.colors as mcolors
num_color = 2
xx, yy = np.meshgrid(np.arange(0,10, step=0.5),
                         np.arange(0,10, step=0.5))
            
Z = np.random.randint(num_color , size=20*20) # Randomly generate integer: 0 or 1
Z= Z.reshape(xx.shape)
colors =  ['sienna','peachpuff']
plt.pcolormesh(xx, yy, Z, cmap  = mcolors.ListedColormap(colors))
plt.savefig('2colormeshgrid.png') 


# %%
num_color = 3
Z = np.random.randint(num_color , size=20*20) # Randomly generate integer: 0 or 1 or 2
Z= Z.reshape(xx.shape)
colors =  ['sienna','peachpuff', 'sandybrown']
plt.pcolormesh(xx, yy, Z, cmap  = mcolors.ListedColormap(colors))
plt.savefig('3colormeshgrid.png') 


# %%



