import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.qmc import LatinHypercube

from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound

import torch
import math

def function(x): # Call function
    return {"obj":np.sin(x)**2+np.sqrt(x+8)}

def plot_function(plot): # Plot function
    X = np.linspace(domain[0],domain[1],100)
    Y = function(X)["obj"]
    plot.plot(X,Y,c="blue",label="Real function (f(x))")
    return X, Y

def fit_gp (train_X, train_Y): # fit a GP
    gp = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        input_transform=Normalize(d=1),
        outcome_transform=Standardize(m=1),
        )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp

def mean_sigma(X): # extract mean and sigma from a posterior GP
    posterior = gp.posterior(X)
    return posterior.mean.detach().numpy().squeeze(-2).squeeze(-1),posterior.stddev.detach().numpy()


domain = [0,5]

"""--------------------- LHS Sampling and evaluation -------------------"""
LHS = LatinHypercube(d=1, seed = 42)
LHS_points = LHS.random(n=4)*domain[1]
train_X = torch.tensor(
    LHS_points
    ).to(torch.double)
train_Y = torch.sin(train_X) ** 2 + torch.sqrt(train_X + 8)


"""--------------------- Fit GP and extract acquisition function -------------------"""
gp = fit_gp(train_X,train_Y)
logEI = LogExpectedImprovement(model=gp, best_f=train_Y.max(),maximize=True)
ucb = UpperConfidenceBound(gp, beta=0.3)

"""--------------------- Define Acq fun for whole domain -------------------"""
x_list = torch.linspace(domain[0],domain[1], 100)
y_mean = [];y_lb = [];y_ub = [];y_ei = [] #define all list at 0
beta = ucb._buffers["beta"]
for x in x_list:
    # Obtain mean and std
    mean, sigma = mean_sigma(torch.tensor([[x]],dtype=torch.double))
    std = torch.sqrt(beta * sigma).detach().numpy()
    # Extract mean, UCB and LCB
    y_mean.append(mean)
    y_lb.append(sum(mean - std))
    y_ub.append(sum(mean + std))
    # Extract EI
    y_ei.append(
        logEI(x.unsqueeze(-1).unsqueeze(-1)).detach().numpy()
    )
y_ei = np.exp(y_ei)
x_max = x_list[np.argmax(y_ub)]

"""--------------------- Extract data for tikz -------------------"""
# LHS points
with open("GP/lhs.dat","w") as f:
    for i,x in enumerate(LHS_points.squeeze(-1)):
        y = train_Y.squeeze(-1)[i].item()
        f.write(f"{x} {y}\n")

# UCB points
with open("GP/ucb.dat","w") as f:
    for i,y in enumerate(y_ub):
        x = x_list[i].item()
        f.write(f"{x} {y}\n")

# LCB points
with open("GP/lcb.dat","w") as f:
    for i,y in enumerate(y_lb):
        x = x_list[i].item()
        f.write(f"{x} {y}\n")

# EI points
with open("GP/ei.dat","w") as f:
    for i,y in enumerate(y_ub):
        x = x_list[i].item()
        f.write(f"{x} {y}\n")

# Mean points
with open("GP/mean.dat","w") as f:
    for i,y in enumerate(y_mean):
        x = x_list[i].item()
        f.write(f"{x} {y}\n")

"""--------------------- Plot Sampling and surrogate -------------------"""
# Plot function and surrogate

fig, ax1 = plt.subplots(figsize = (12,6))
X, Y = plot_function(ax1)
ax1.scatter(train_X,train_Y,label =  "initial sampling", c="blue")
ax1.set_ylim(2.5,5)

# Plot Mean,UCB...
ax1.plot(x_list,y_mean,c="black",label="Kernel mean function", linestyle = "dashed")
ax1.fill_between(x_list,y_lb,y_ub,alpha=0.2, color = "red")
ax1.plot(x_list,y_lb, linestyle = "dashed",color = "red",alpha = 0.3)
ax1.plot(x_list,y_ub, linestyle = "dashed",color = "red",alpha = 0.3, label = "Confidence interval")

# Plot EI
ax2 = ax1.twinx()
ax2.plot(x_list, y_ei)
ax2.set_ylim(0,max(y_ei)*2)

# Plot x_max
""" ax1.scatter(x_max,max(y_ub), label="x_max of UCB", c="red", marker="x")
ax1.scatter(x_max,function(x_max.item())["obj"], c="blue", marker="x", label = "evaluation of best point f(x_max)")
ax1.vlines(x_max,min(Y),4.8,colors="red", alpha=0.5) """



# Last command
ax1.legend(loc = "upper left")
ax1.set_xlim(0,5)
ax1.set_ylabel("f(x)")
ax2.set_ylabel("Expected Improvement")
fig.suptitle("Surrogate of f(x) using Gaussian Process")
fig.tight_layout()
plt.savefig("GP/gaussian whole process")

