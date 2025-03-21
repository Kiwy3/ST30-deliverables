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
saving = True
folder = "../tikz_picture/BaMSOO/"
def function(x): # Call function
    return {"obj":-np.sin(1.1*x)**3+np.sqrt(x+7)}

def plot_function(plot): # Plot function
    X = np.linspace(domain[0],domain[1],100)
    Y = function(X)["obj"]
    plot.plot(X,Y,c="blue",label="Real function (f(x))")
    return X, Y

def plot(): # Plot function
    # Plot function and surrogate

    fig, ax1 = plt.subplots(figsize = (12,6))
    X, Y = plot_function(ax1)
    ax1.scatter(train_X,train_Y,label =  "initial sampling", c="blue")
    #ax1.set_ylim(2.5,5)

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
    ax1.scatter(x_max,max(y_ub), label="x_max of UCB", c="red", marker="x")
    ax1.scatter(x_max,function(x_max.item())["obj"], c="blue", marker="x", label = "evaluation of best point f(x_max)")
    ax1.vlines(x_max,min(Y),4.8,colors="red", alpha=0.5)



    # Last command
    ax1.legend(loc = "upper left")
    ax1.set_xlim(domain[0],domain[1])
    #ax1.set_xlim(2,4)

    ax1.set_ylabel("f(x)")
    ax2.set_ylabel("Expected Improvement")
    fig.suptitle("Surrogate of f(x) using Gaussian Process")
    fig.tight_layout()
    plt.show()
    #plt.savefig("GP/gaussian whole process")"""

def fit_gp (train_X, train_Y): # fit a GP
    Y_var = torch.zeros_like(train_Y)
    gp = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        train_Yvar=Y_var,
        input_transform=Normalize(d=1),
        outcome_transform=Standardize(m=1),
        )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp

def mean_sigma(X): # extract mean and sigma from a posterior GP
    posterior = gp.posterior(X)
    return posterior.mean.detach().numpy().squeeze(-2).squeeze(-1),posterior.stddev.detach().numpy()


domain = [0,6]

"""--------------------- LHS Sampling and evaluation -------------------"""
points = [3]+[1,5]#+[4.3,5.6]
train_X = torch.tensor(
    points
    ).to(torch.double).unsqueeze(-1)
train_Y = function(train_X)["obj"]

"""--------------------- Fit GP and extract acquisition function -------------------"""
gp = fit_gp(train_X,train_Y)
logEI = LogExpectedImprovement(model=gp, best_f=train_Y.max(),maximize=True)
ucb = UpperConfidenceBound(gp, beta=1.96)

"""--------------------- Define Acq fun for whole domain -------------------"""
x_list = torch.linspace(domain[0],domain[1], 100)
y_mean = [];y_lb = [];y_ub = [];y_ei = [] 
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

plot()

"""--------------------- Extract data for tikz -------------------"""
if saving : 
    # LHS points
    with open(f"{folder}soo_1.dat","w") as f:
        for i,x in enumerate(points):
            y = train_Y.squeeze(-1)[i].item()
            f.write(f"{x} {y}\n")

    # UCB points
    with open(f"{folder}ucb_1.dat","w") as f:
        for i,y in enumerate(y_ub):
            x = x_list[i].item()
            f.write(f"{x} {y}\n")


"""--------------------- LHS Sampling and evaluation -------------------"""
points = [3]+[1,5]+[4.3,5.6]
train_X = torch.tensor(
    points
    ).to(torch.double).unsqueeze(-1)
train_Y = function(train_X)["obj"]

"""--------------------- Fit GP and extract acquisition function -------------------"""
gp = fit_gp(train_X,train_Y)
logEI = LogExpectedImprovement(model=gp, best_f=train_Y.max(),maximize=True)
ucb = UpperConfidenceBound(gp, beta=1)

"""--------------------- Define Acq fun for whole domain -------------------"""
x_list = torch.linspace(domain[0],domain[1], 100)
y_mean = [];y_lb = [];y_ub = [];y_ei = [] 
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
plot()
"""--------------------- Extract data for tikz -------------------"""
if saving : 
    # LHS points
    with open(f"{folder}soo_2.dat","w") as f:
        for i,x in enumerate(points):
            y = train_Y.squeeze(-1)[i].item()
            f.write(f"{x} {y}\n")

    # UCB points
    with open(f"{folder}ucb_2.dat","w") as f:
        for i,y in enumerate(y_ub):
            x = x_list[i].item()
            f.write(f"{x} {y}\n")

if saving : 
    """ Obtain LCB for useful points """
    points = [2.3,3.6]
    lb_list = []
    for x in points:
        mean, sigma = mean_sigma(torch.tensor([[x]],dtype=torch.double))
        std = torch.sqrt(beta * sigma).detach().numpy()
        lb = mean - std
        lb_list.append(lb.item())

    with open(f"{folder}lcb_2.dat","w") as f:
        for i in range(len(points)):
            x = points[i]
            y = lb_list[i]
            f.write(f"{x} {y}\n")
