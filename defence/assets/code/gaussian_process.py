import numpy as np
import matplotlib.pyplot as plt

# 1. Kernel Function (RBF Kernel)
def rbf_kernel(x1, x2, length_scale=1.0, sigma_f=1.0):
    """Radial Basis Function (RBF) Kernel."""
    sqdist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    print(sqdist.shape)
    return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)

# 2. Generate Synthetic Data
np.random.seed(42)
X_train = np.array([[1], [3], [5], [6], [8]])
y_train = np.sin(X_train).ravel() + np.random.normal(0, 0.1, X_train.shape[0])

# 3. Define Test Points
X_test = np.linspace(0, 10, 50).reshape(-1, 1)

# 4. Compute the Covariance Matrices
K = rbf_kernel(X_train, X_train) + 1e-8 * np.eye(len(X_train))  # Add small noise for numerical stability
K_s = rbf_kernel(X_train, X_test)
K_ss = rbf_kernel(X_test, X_test) + 1e-8 * np.eye(len(X_test))

# 5. Compute the Posterior Mean and Covariance
K_inv = np.linalg.inv(K)

# Posterior mean
mu_s = K_s.T @ K_inv @ y_train

# Posterior covariance
cov_s = K_ss - K_s.T @ K_inv @ K_s
std_s = np.sqrt(np.diag(cov_s))

# 6. Visualization
plt.figure(figsize=(10, 6))
plt.plot(X_test, np.sin(X_test), 'r--', label='True function (sin(x))')
plt.scatter(X_train, y_train, c='black', label='Observed Data')
plt.plot(X_test, mu_s, 'b-', label='Posterior Mean')
plt.fill_between(X_test.ravel(),
                 mu_s - 1.96 * std_s,
                 mu_s + 1.96 * std_s,
                 alpha=0.2, color='blue', label='95% Confidence Interval')

plt.title('Gaussian Process Regression from Scratch')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()

# 7 Save data
saving = True
if saving :  
    X_test = X_test.squeeze(1)
    X_train = X_train.squeeze(1)
    folder = "../tikz_picture/gaussian_process/"
    # LHS points
    with open(f"{folder}lhs.dat","w") as f:
        for i,x in enumerate(X_train):
            y = y_train[i]
            f.write(f"{x} {y}\n")

    # Mean points
    with open(f"{folder}mean.dat","w") as f:
        for i in range(len(X_test)):
            x = X_test[i]
            y = mu_s[i]
            f.write(f"{x} {y}\n")

    # UCB points
    with open(f"{folder}ucb.dat","w") as f:
        for i in range(len(X_test)):
            x = X_test[i]
            y = mu_s[i] + 1.96 * std_s[i]
            f.write(f"{x} {y}\n")

    # LCB points
    with open(f"{folder}lcb.dat","w") as f:
        for i in range(len(X_test)):
            x = X_test[i]
            y = mu_s[i] - 1.96 * std_s[i]
            f.write(f"{x} {y}\n")

