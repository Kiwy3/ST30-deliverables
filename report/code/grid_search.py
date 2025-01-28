def himmelblau(x) :
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


import matplotlib.pyplot as plt
import numpy as np

def himmel_plot():
    fig,ax = plt.subplots()
    x = np.linspace(-5,5,200)
    X,Y = np.meshgrid(x,x)
    Z = himmelblau([X,Y])

    map = ax.contourf(X,Y,Z,cmap="Blues",levels=50,alpha=0.8,vmin=-50,vmax=400)
    fig.colorbar(map)

    #plt.title("Grid Search on Himmelblau function")

himmel_plot()

n = 25
n_sq = int(np.sqrt(n))
x1 = np.linspace(-4.5,4.5,n_sq)
x2 = np.linspace(-4.5,4.5,n_sq)

X1, X2 = np.meshgrid(x1,x2)
plt.scatter(X1,X2,s=10, c="red",marker='o')
plt.tight_layout()
plt.savefig("assets/img/chap_2/plots/grid_search.png")




