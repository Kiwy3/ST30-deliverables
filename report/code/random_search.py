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

    #plt.title("Random Search on Himmelblau function")

himmel_plot()

n = 25

x1 = np.random.uniform(-5,5,n)
x2 = np.random.uniform(-5,5,n)
plt.scatter(x1,x2,s=10, c="red",marker='o')
plt.tight_layout()
plt.savefig("assets/img/chap_2/plots/random_search.png")




