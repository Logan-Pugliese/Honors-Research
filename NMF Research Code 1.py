# %% [markdown]
# Creating NMF Algo

# %%
##NMF
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
from sklearn.decomposition import NMF
import pandas as pd
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# %%
def random_init(V, rank):
    """
    Randomly initializes the factor matrices W and H for NMF.

    Parameters:
    - V: Input matrix (m x n)
    - rank: Rank of the factorization

    Returns:
    - W: Initialized W matrix (m x rank)
    - H: Initialized H matrix (rank x n)
    """
    # Get the dimensions of the input matrix V
    num_docs = V.shape[0]  # Number of rows (documents or samples)
    num_terms = V.shape[1]  # Number of columns (terms or features)
    
    # Initialize W and H with random values in the range [0, 1]
    W = np.random.rand(num_docs, rank)  # W is (m x rank)
    H = np.random.rand(rank, num_terms)  # H is (rank x n)
    
    return W, H


# Function to compute the NNDSVD initialization
def nndsvd_init(A, k, variant='basic'):
    """
    A: Input data matrix (nonnegative)
    k: Number of factors
    variant: 'basic', 'a', or 'ar' for different NNDSVD variants
    """
    # Compute the largest k singular triplets of A
    U, S, Vt = svds(A, k=k)
    
    # Initialize W and H matrices
    W = np.zeros((A.shape[0], k))
    H = np.zeros((k, A.shape[1]))
    
    # Initialize the first column
    W[:, 0] = np.sqrt(S[-1]) * U[:, -1]
    H[0, :] = np.sqrt(S[-1]) * Vt[-1, :]
    
    # For each singular triplet, compute positive and negative sections
    for j in range(1, k):
        x = U[:, -(j+1)]
        y = Vt[-(j+1), :]
        
        # Positive and negative sections
        xp, xn = np.maximum(x, 0), -np.minimum(x, 0)
        yp, yn = np.maximum(y, 0), -np.minimum(y, 0)
        
        # Norms of positive and negative sections
        xpnrm, ypnrm = np.linalg.norm(xp), np.linalg.norm(yp)
        xnnrm, ynnrm = np.linalg.norm(xn), np.linalg.norm(yn)
        
        mp, mn = xpnrm * ypnrm, xnnrm * ynnrm
        
        if mp > mn:
            u = xp / xpnrm
            v = yp / ypnrm
            sigma = mp
        else:
            u = xn / xnnrm
            v = yn / ynnrm
            sigma = mn
        
        W[:, j] = np.sqrt(S[-(j+1)] * sigma) * u
        H[j, :] = np.sqrt(S[-(j+1)] * sigma) * v
    
    # Apply the NNDSVDa or NNDSVDar variants if specified
    if variant == 'a':
        # Replace zero values with the mean of matrix A
        mean_A = np.mean(A)
        W[W == 0] = mean_A
        H[H == 0] = mean_A
        
    elif variant == 'ar':
        # Replace zero values with small random values in [0, mean(A) / 100]
        mean_A = np.mean(A)
        W[W == 0] = np.random.uniform(0, mean_A / 100, np.count_nonzero(W == 0))
        H[H == 0] = np.random.uniform(0, mean_A / 100, np.count_nonzero(H == 0))
    
    return W, H

# %%
def mu(W, H, V, max_iter):
    """
    Parameters:
    - W: Randomly Initialized W (m x k)
    - H: Randomly Initialized H (k x n)
    - V: Input matrix (m x n)
    - max_iter: Maximum number of iterations

    Returns:
    - W: Factorized matrix W
    - H: Factorized matrix H
    - norms: List of Frobenius norms at each iteration
    """
    
    norms = []
    epsilon = 1.0e-10  # Small constant to avoid division by zero
    
    for _ in range(max_iter):
        # Update H
        W_TV = W.T @ V  # W.T * V
        W_TWH = W.T @ W @ H + epsilon  # W.T * W * H + epsilon (element-wise)
        H = H * (W_TV / W_TWH)  # Element-wise division and update
        
        # Update W
        VH_T = V @ H.T  # V * H.T
        WHH_T = W @ H @ H.T + epsilon  # W * H * H.T + epsilon (element-wise)
        W = W * (VH_T / WHH_T)  # Element-wise division and update
       
        # Calculate Frobenius norm and append it to the list
        norm = np.linalg.norm(V - W @ H, 'fro')
        norms.append(norm)
        
    return W, H, norms


# %%
@ignore_warnings(category=ConvergenceWarning)
def test_nmf_with_sklearn(V, rank, W, H, max_iter, solver):
    """
    Test the NMF algorithm using scikit-learn with custom random initialization.
    
    Parameters:
    - V: Input matrix (non-negative) (m x n)
    - rank: Rank for the factorization
    - W: Initial W guess (needed for custom implementation)
    - H: Initial H guess (needed for custom implementation)
    - max_iter: Maximum number of iterations for the NMF algorithm
    - Solver: solver used by Sklearn (either 'mu' or 'cd')
    
    Returns:
    - W: Factorized matrix W
    - H: Factorized matrix H
    - norms: List of Frobenius norms at each iteration
    """
    # Random initialization of W and H
    model = NMF(n_components=rank, init='custom', max_iter=1, solver=solver, random_state=42)

    # Initial fit with one iteration to get initial W, H
    W = model.fit_transform(V, W=W, H=H)
    H = model.components_

    # Initialize list to store norms
    norms = []
    norms.append(np.linalg.norm(V - np.dot(W, H), 'fro'))
    # Iterate for max_iter steps
    for iteration in range(max_iter):
        # Perform one iteration of NMF
        W = model.fit_transform(V, W=W, H=H)  # Perform update for W
        H = model.components_  # Perform update for H

        # Calculate the Frobenius norm of the reconstruction error
        norm = np.linalg.norm(V - np.dot(W, H), 'fro')
        norms.append(norm)
    
    # Return final W, H, and list of norms over the iterations
    return W, H, norms

# %%
def hals_nmf(Y, A, X, max_iter):
    """HALS algorithm for Non-negative Matrix Factorization (NMF) with error tracking
    
    Args:
        Y: Input data matrix (I x K)
        A: Initial guess for the basis matrix (I x J)
        X: Initial guess for the component matrix (J x K)
        max_iter: Maximum number of iterations
        
    Returns:
        A: Updated basis matrix
        X: Updated component matrix
        errors: List of reconstruction errors over iterations
    """
    I, K = Y.shape
    J = A.shape[1]
    
    errors = []  # List to track reconstruction errors
    epsilon=1e-10
    # Normalize initial matrices (columns of A)
    A = A / np.linalg.norm(A, axis=0, keepdims=True)
    
    for iteration in range(max_iter):
        # Update X
        for j in range(J):
            A_j = A[:, j]
            # Update X_j by fixing other columns of X
            residual = Y - np.dot(A, X) + np.outer(A_j, X[j, :])
            denominator = np.dot(A_j.T, A_j) + epsilon  # Add epsilon to avoid division by zero
            X_j_new = np.dot(A_j.T, residual) / denominator
            X[j, :] = np.maximum(X_j_new, 0)  # Ensure non-negativity
        
        # Update A
        for j in range(J):
            X_j = X[j, :]
            # Update A_j by fixing other columns of A
            residual = Y - np.dot(A, X) + np.outer(A[:, j], X_j)
            denominator = np.dot(X_j, X_j.T) + epsilon  # Add epsilon to avoid division by zero
            A_j_new = np.dot(residual, X_j.T) / denominator
            A[:, j] = np.maximum(A_j_new, 0)  # Ensure non-negativity

            # Normalize A_j if its norm is greater than epsilon
            norm_A_j = np.linalg.norm(A[:, j], 2)
            if norm_A_j > epsilon:
                A[:, j] = A[:, j] / norm_A_j
        
        # Compute residual and track error
        E = Y - np.dot(A, X)
        error = np.linalg.norm(E, 'fro')
        errors.append(error)
    
    return A, X, errors


# %%
def coord_descent_nmf(X, W_init, H_init, max_iter):
    W, H = W_init.copy(), H_init.copy()
    norms = []
    for iteration in range(max_iter):
        # Update H
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                H[i, j] = max(0, H[i, j] - (W[:, i] @ (W @ H - X)[:, j]) / (W[:, i].T @ W[:, i]))
        
        # Update W
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] = max(0, W[i, j] - ((W @ H - X) @ H.T)[i, j] / (H[j, :] @ H[j, :].T))
        
        # Calculate Frobenius norm of reconstruction error
        norms.append(np.linalg.norm(X - W @ H, 'fro'))
    
    return W, H, norms

# %%
def bcd_nmf(X, W_init, H_init, max_iter):
    W, H = W_init.copy(), H_init.copy()
    norms = []
    for iteration in range(max_iter):
        # Update H using least squares
        H = np.linalg.lstsq(W, X, rcond=None)[0]
        H = np.maximum(H, 0)  # Ensure non-negative constraint
        
        # Update W using least squares
        W = np.linalg.lstsq(H.T, X.T, rcond=None)[0].T
        W = np.maximum(W, 0)  # Ensure non-negative constraint
        
        # Calculate Frobenius norm of reconstruction error
        norms.append(np.linalg.norm(X - W @ H, 'fro'))
    
    return W, H, norms

# %%
# Ensure the seed is set for reproducibility
np.random.seed(42)

# Parameters
max_iters = 200
rank = 4

# Assume X_hat is the input matrix you want to factorize
##Initialization
W_0 = np.random.rand(20, 20)
H_0 = np.random.rand(20, 20)
X_hat = W_0 @ H_0

# Initialization methods to test
init_methods = ['random', 'nndsvd', 'nndsvda', 'nndsvdar']

# Store results for plotting later
results = {}
final_norms = []  # To store the final norms for the table

# Loop through each initialization method and each NMF algorithm (MU and Sklearn)
for init_method in init_methods:
    
    # Initialize W and H based on the method
    if init_method == 'random':
        W_init, H_init = random_init(X_hat, rank)
    elif init_method == 'nndsvd':
        W_init, H_init = nndsvd_init(X_hat, rank, variant='basic')
    elif init_method == 'nndsvda':
        W_init, H_init = nndsvd_init(X_hat, rank, variant='a')
    elif init_method == 'nndsvdar':
        W_init, H_init = nndsvd_init(X_hat, rank, variant='ar')
    
    # Multiplicative Update NMF (custom implementation)
    W_mu, H_mu, norms_mu = mu(W_init, H_init, X_hat, max_iters)
    
    # Store results for plotting
    results[f'{init_method}_mu'] = norms_mu
    
    # Sklearn NMF with custom initialization
    W_sklearn, H_sklearn, norms_sklearn_mu = test_nmf_with_sklearn(X_hat, rank, W_init, H_init, max_iters, solver='mu')
    
    # Store results for plotting
    results[f'{init_method}_sklearn_mu'] = norms_sklearn_mu
    
    # Sklearn NMF with custom initialization using Coordinate Descent solver
    W_sklearn_cd, H_sklearn_cd, norms_sklearn_cd = test_nmf_with_sklearn(X_hat, rank, W_init, H_init, max_iters, solver='cd')
    
    # Store results for plotting
    results[f'{init_method}_sklearn_cd'] = norms_sklearn_cd
    
    # HALS NMF
    W_hals, H_hals, norms_hals = hals_nmf(X_hat, W_init, H_init, max_iter=max_iters)
    
    # Store results for plotting
    results[f'{init_method}_hals'] = norms_hals
    
    # Coordinate Descent NMF (Paper 2)
    W_coord, H_coord, norms_coord = coord_descent_nmf(X_hat, W_init, H_init, max_iters)
    results[f'{init_method}_coord_descent'] = norms_coord
    
    # Block Coordinate Descent NMF (Paper 3)
    W_bcd, H_bcd, norms_bcd = bcd_nmf(X_hat, W_init, H_init, max_iters)
    results[f'{init_method}_bcd'] = norms_bcd
    
    # Append final Frobenius norms to the list for the table
    final_norms.append([
        init_method, 
        norms_mu[-1], 
        norms_sklearn_mu[-1], 
        norms_sklearn_cd[-1], 
        norms_hals[-1],
        norms_coord[-1],
        norms_bcd[-1]
    ])

# Create a DataFrame to display the final norms
df_final_norms = pd.DataFrame(final_norms, columns=[
    'Initialization Method', 'Final Norm (MU)', 'Final Norm (Sklearn, MU)', 
    'Final Norm (Sklearn, CD)', 'Final Norm (HALS)', 'Final Norm (Coord Descent)', 
    'Final Norm (BCD)'
])

# Display the table
print("\nFinal Frobenius Norms Table:")
print(df_final_norms)

# Plot the results
fig, axs = plt.subplots(2, 4, figsize=(20, 12))

# Plot loss for Multiplicative Update (MU)
for init_method in init_methods:
    axs[0, 0].plot(range(len(results[f'{init_method}_mu'])), results[f'{init_method}_mu'], label=f'{init_method} Init (MU)')
axs[0, 0].legend()
axs[0, 0].set_title('Multiplicative Update NMF Loss')
axs[0, 0].set_xlabel('Iterations')
axs[0, 0].set_ylabel('Frobenius Norm')

# Plot loss for Sklearn NMF (Multiplicative Update)
for init_method in init_methods:
    axs[0, 1].plot(range(len(results[f'{init_method}_sklearn_mu'])), results[f'{init_method}_sklearn_mu'], label=f'{init_method} Init (Sklearn MU)')
axs[0, 1].legend()
axs[0, 1].set_title('Sklearn NMF Loss (MU Solver)')
axs[0, 1].set_xlabel('Iterations')
axs[0, 1].set_ylabel('Frobenius Norm')

# Plot loss for Sklearn NMF (Coordinate Descent)
for init_method in init_methods:
    axs[0, 2].plot(range(len(results[f'{init_method}_sklearn_cd'])), results[f'{init_method}_sklearn_cd'], label=f'{init_method} Init (Sklearn CD)')
axs[0, 2].legend()
axs[0, 2].set_title('Sklearn NMF Loss (CD Solver)')
axs[0, 2].set_xlabel('Iterations')
axs[0, 2].set_ylabel('Frobenius Norm')

# Plot loss for HALS NMF
for init_method in init_methods:
    axs[0, 3].plot(range(len(results[f'{init_method}_hals'])), results[f'{init_method}_hals'], label=f'{init_method} Init (HALS)')
axs[0, 3].legend()
axs[0, 3].set_title('HALS NMF Loss')
axs[0, 3].set_xlabel('Iterations')
axs[0, 3].set_ylabel('Frobenius Norm')

# Plot loss for Coordinate Descent NMF
for init_method in init_methods:
    axs[1, 0].plot(range(len(results[f'{init_method}_coord_descent'])), results[f'{init_method}_coord_descent'], label=f'{init_method} Init (Coord Descent)')
axs[1, 0].legend()
axs[1, 0].set_title('Coordinate Descent NMF Loss')
axs[1, 0].set_xlabel('Iterations')
axs[1, 0].set_ylabel('Frobenius Norm')

# Plot loss for Block Coordinate Descent NMF
for init_method in init_methods:
    axs[1, 1].plot(range(len(results[f'{init_method}_bcd'])), results[f'{init_method}_bcd'], label=f'{init_method} Init (BCD)')
axs[1, 1].legend()
axs[1, 1].set_title('Block Coordinate Descent NMF Loss')
axs[1, 1].set_xlabel('Iterations')
axs[1, 1].set_ylabel('Frobenius Norm')

plt.tight_layout()
plt.show()


