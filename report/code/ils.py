from numpy import asarray
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi
from numpy.random import randn
from numpy.random import rand
from numpy.random import seed
import matplotlib.pyplot as plt
import numpy as np
 
# objective function
def objective(v):
    x, y = v
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
 

# check if a point is within the bounds of the search
def in_bounds(point, bounds):
   # enumerate all dimensions of the point
   for d in range(len(bounds)):
      # check if out of bounds for this dimension
      if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
         return False
   return True
 
# hill climbing local search algorithm
def hillclimbing(objective, bounds, n_iterations, step_size, start_pt):
   x1 = [ ]
   x2 = [ ]

   k1 = [ ]
   k2 = [ ]
   # store the initial point
   solution = start_pt
   # evaluate the initial point
   solution_eval = objective(solution)
   # run the hill climb
   for i in range(n_iterations):
      # take a step
      candidate = None
      while candidate is None or not in_bounds(candidate, bounds):
         candidate = solution + randn(len(bounds)) * step_size
      # evaluate candidate point
      candidte_eval = objective(candidate)
      # check if we should keep the new point
      if candidte_eval <= solution_eval:
        x1.append(candidate[0])
        x2.append(candidate[1])

         # store the new point
        solution, solution_eval = candidate, candidte_eval
      else : 
        k1.append(candidate[0])
        k2.append(candidate[1])
   return [solution, solution_eval, (x1,x2), (k1,k2)]
 
# iterated local search algorithm
def iterated_local_search(objective, bounds, n_iter, step_size, n_restarts, p_size):   
  history = []
  unpromising_history = []
  # define starting point
  best = None
  while best is None or not in_bounds(best, bounds):
    best = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
  best = [0,0]
  # evaluate current best point
  best_eval = objective(best)
  # enumerate restarts
  for n in range(n_restarts):
    # generate an initial point as a perturbed version of the last best
    start_pt = None
    while start_pt is None or not in_bounds(start_pt, bounds):
      start_pt = best + randn(len(bounds)) * p_size
    # perform a stochastic hill climbing search
    solution, solution_eval, hill_history, unpromising = hillclimbing(objective, bounds, n_iter, step_size, start_pt)
    history.append(hill_history)
    unpromising_history.append(unpromising)
    # check for new best
    if solution_eval < best_eval:
      best, best_eval = solution, solution_eval
      print('Restart %d, best: f(%s) = %.5f' % (n, best, best_eval))
  return [best, best_eval, history, unpromising_history]
 

def himmel_plot():
    fig,ax = plt.subplots()
    x = np.linspace(-1,4,200)
    X,Y = np.meshgrid(x,x)
    Z = objective([X,Y])

    map = ax.contourf(X,Y,Z,cmap="Blues",levels=50,alpha=0.8,vmin=-50,vmax=400)
    fig.colorbar(map)

    #plt.title("Iterated Local Search on Himmelblau function")
# seed the pseudorandom number generator
seed(73754615)
# define range for input
bounds = asarray([[-4.0, 4.0], [-4.0, 4.0]])
# define the total iterations
n_iter = 8
# define the maximum step size
step_size = 0.4
# total number of random restarts
n_restarts = 4
# perform the hill climbing search
best, score,history, unpromising= iterated_local_search(objective, bounds, n_iter, step_size, n_restarts=n_restarts, p_size=0.1)
print('Done!')
print('f(%s) = %f' % (best, score))

himmel_plot()

for i in range(len(history)):
    x1,x2 = history[i]
    plt.scatter(x1,x2,s=10, label = f"local search {i}")
    for i in range(len(x1)):
      if i%2 == 0:
         step = 0.05
      else :
         step = -0.05
      plt.text(x1[i]+step,x2[i]+0.05,str(i))

k1= []
k2 = []
for i in range(len(unpromising)):
   for j in range(len(unpromising[i])):
      k1.append(unpromising[i][0][j])
      k2.append(unpromising[i][1][j])
plt.scatter(k1,k2,s=6, c="black", label = "Unpromising")
plt.legend()
plt.tight_layout()
plt.savefig("assets/img/chap_2/plots/ils.png")