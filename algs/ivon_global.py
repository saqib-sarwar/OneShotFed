
# import torch
# import copy
# from utils.compute_accuracy import test_img
# from torch.nn.utils import parameters_to_vector, vector_to_parameters

# from ivon import IVON

import numpy as np
from scipy.linalg import sqrtm

def wasserstein_distance(mu, Sigma, mu_i, Sigma_i):
    """
    Calculate the 2-Wasserstein distance between two Gaussian distributions.
    Parameters:
        mu: Mean of the first Gaussian (global mean).
        Sigma: Covariance matrix of the first Gaussian (global covariance).
        mu_i: Mean of the i-th Gaussian (local mean).
        Sigma_i: Covariance matrix of the i-th Gaussian (local covariance).
    Returns:
        W2: Wasserstein distance between the two Gaussians.
    """
    mean_diff = np.linalg.norm(mu - mu_i)
    cov_term = np.trace(Sigma + Sigma_i - 2 * sqrtm(sqrtm(Sigma).dot(Sigma_i).dot(sqrtm(Sigma))))
    W2 = mean_diff**2 + cov_term
    return W2

def optimize_wasserstein_barycenter(local_means, local_covariances, weights, max_iters=100, tol=1e-6):
    """
    Optimize to find the Wasserstein barycenter (global Gaussian) for N local Gaussian models.
    Parameters:
        local_means: List of mean vectors for each local model (shape: N x D).
        local_covariances: List of covariance matrices for each local model (shape: N x D x D).
        weights: List of weights (scalars) for each local model.
        max_iters: Maximum number of iterations.
        tol: Convergence tolerance.
    Returns:
        mu_global: Optimized mean for the global model.
        Sigma_global: Optimized covariance for the global model.
    """
    # Initialization with weighted average of local means
    mu_global = np.average(local_means, axis=0, weights=weights)
    Sigma_global = np.average(local_covariances, axis=0, weights=weights)
    
    for _ in range(max_iters):
        Sigma_global_sqrt = sqrtm(Sigma_global)
        Sigma_updates = np.zeros_like(Sigma_global)

        for i, (mu_i, Sigma_i, weight) in enumerate(zip(local_means, local_covariances, weights)):
            # Calculate transport matrix for covariance alignment
            T_i = sqrtm(Sigma_global_sqrt @ Sigma_i @ Sigma_global_sqrt)
            Sigma_updates += weight * T_i

        # Update the global covariance to the weighted average of transport matrices
        Sigma_new = Sigma_updates @ Sigma_updates.T
        mu_new = np.average(local_means, axis=0, weights=weights)
        
        # Check for convergence
        if np.linalg.norm(mu_global - mu_new) < tol and np.linalg.norm(Sigma_global - Sigma_new) < tol:
            break
        
        mu_global = mu_new
        Sigma_global = Sigma_new

    return mu_global, Sigma_global

# Sample data
N = 5  # Number of local models
D = 3  # Dimensionality of each mean vector
np.random.seed(0)

local_means = [np.random.rand(D) for _ in range(N)]
local_covariances = [np.eye(D) + 0.1 * np.random.rand(D, D) for _ in range(N)]
weights = [1/N] * N  # Equal weights

# Compute the Wasserstein barycenter
mu_global, Sigma_global = optimize_wasserstein_barycenter(local_means, local_covariances, weights)
print("Global Mean:\n", mu_global)
print("Global Covariance:\n", Sigma_global)



# def one_shot_ivon(net_glob, hessian_sum, grad_avg, p, dataset_val, args_ivon, args):

#     net_glob_copy = copy.deepcopy(net_glob)
#     w_avg = parameters_to_vector(net_glob.parameters())
#     w = parameters_to_vector(net_glob.parameters())
#     T = args_ivon['T']
#     eta = args_ivon['eta']
#     test_acc_i_max = 0

#     with torch.no_grad():

#         test_acc_tracker = []
#         test_acc_i_max = 0

#         # Initialize Adam optimizer using net_glob's parameters
#         optimizer =  IVON(net_glob.parameters(), 
#                             lr=eta,
#                             ess=args['ess'], 
#                             weight_decay=args['weight_decay'], 
#                             hess_init=args['hess_init'])

#         for k in range(T):
#         # Calculate the gradient approximation `v`
#             v = hessian_sum * w - grad_avg
            
#             # Zero the gradients
#             optimizer.zero_grad()
            
#             # Set gradients to `v`
#             vector_to_parameters(v, net_glob.parameters())  # Setting `v` as the gradients
#             for param in net_glob.parameters():
#                 param.grad = param.data
            
#             for _ in range(args['mc_train']):
#                 with optimizer.sampled_params(train=True):
#                     pass
            
#             # Perform optimization step using Adam
#             optimizer.step()

#             # Get the updated weight vector
#             w = parameters_to_vector(net_glob.parameters())

#             if(k%100==0):

#                 w_vec_estimate = w
#                 vector_to_parameters(w_vec_estimate,net_glob_copy.parameters())
#                 test_acc_i, test_loss_i = test_img(net_glob_copy, dataset_val, args)
#                 if(test_acc_i > test_acc_i_max):
#                     test_acc_i_max = test_acc_i
#                     best_parameters = w

#                 print ("Val Test Acc: ", test_acc_i, " Val Test Loss: ", test_loss_i)
#                 test_acc_tracker.append(test_acc_i)

#     return best_parameters
