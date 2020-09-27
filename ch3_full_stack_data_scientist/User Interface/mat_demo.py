import io
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, flash, redirect

app = Flask(__name__)

@app.route('/')
def index():
    # generate plot
    x = np.linspace(0, 5, 20) # 20 points with X−axis range from 0 to 5
    i = 0
    for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
        plt.plot(x[i],y[i], marker, label="marker='{0}'".format(marker))
        i = i + 1
    plt.legend(loc='best',fontsize=9)
    plt.title('This is a scatterplot' , fontsize = 14)
    plt.xlabel('This is X−values', fontsize = 12)
    plt.ylabel('This is Y−values' , fontsize = 12)
    plt.savefig('ascatter.png')
            
    # render the template with the plot
    return render_template('index_mat.html', name = "Scatter Plot", path = 'ascatter.png')

######section 3#######
if __name__ == "__main__":
    app.run()