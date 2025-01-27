import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition.analytic import LogExpectedImprovement, UpperConfidenceBound

import torch
import math

def function(x):
    print(x)
    return {"obj":np.sin(x)**3+np.sqrt(x+8)}

X = np.linspace(-8,8,100)
Y = function(X)["obj"]
plt.figure(figsize=(12,6))
plt.plot(X,Y,c="blue",label="Real function (f(x))")

gen = torch.manual_seed(4654)
#train_X = torch.rand(5,1, generator=gen,dtype=torch.double)*16-8
train_X = torch.linspace(-7, 7, 7).unsqueeze(-1).to(torch.double)
print(train_X)
train_Y = torch.sin(train_X) ** 3 + torch.sqrt(train_X + 8)
#Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
plt.scatter(train_X,train_Y,label =  "initial sampling", c="blue")

lower_bounds = torch.tensor(X.min())
upper_bounds = torch.tensor(X.max())
bounds = torch.stack((lower_bounds, upper_bounds)).unsqueeze(-1).to(torch.double)


gp = SingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    input_transform=Normalize(d=1),
    outcome_transform=Standardize(m=1),
    )
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

logEI = LogExpectedImprovement(model=gp, best_f=Y.max(),maximize=True)
ucb = UpperConfidenceBound(gp, beta=0.2)

def mean_sigma(X):
    posterior = gp.posterior(X)
    return posterior.mean.detach().numpy(),posterior.stddev.detach().numpy()

x_list = torch.linspace(-8, 8, 500)
y_mean = []
y_lb = []
y_ub = []
beta = ucb._buffers["beta"]
for x in x_list:
    mean, sigma = mean_sigma(torch.tensor([[x]],dtype=torch.double))
    mean = mean.squeeze(-2).squeeze(-1)
    y_mean.append(mean)
    
    std = torch.sqrt(beta * sigma).detach().numpy()
    y_lb.append(sum(mean - std))
    y_ub.append(sum(mean + std))

x_max = x_list[np.argmax(y_ub)]
plt.scatter(x_max,max(y_ub), label="x_max of UCB", c="red", marker="x")
plt.scatter(x_max,function(x_max.item())["obj"], c="blue", marker="x", label = "evaluation of best point f(x_max)")
plt.vlines(x_max,min(Y),4.8,colors="red", alpha=0.5)

plt.plot(x_list,y_mean,c="black",label="Kernel mean function", linestyle = "dashed")
plt.fill_between(x_list,y_lb,y_ub,alpha=0.2,label="uncertainty", color = "red")
plt.plot(x_list,y_lb, linestyle = "dashed",color = "red",alpha = 0.3)
plt.plot(x_list,y_ub, linestyle = "dashed",color = "red",alpha = 0.3, label = "Confidence interval")
plt.legend()
#plt.title("Surrogate of f(x) using Gaussian Process")
plt.xlabel("x")
plt.ylabel("f(x)=sin(x)^3+sqrt(x+5)")
plt.savefig("../assets/img/chap_2/plots/gaussian_process.png")






