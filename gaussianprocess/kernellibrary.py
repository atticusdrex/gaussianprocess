import jax.numpy as jnp 

# Define the kernel function (e.g., RBF kernel)
def rbf_kernel(x1, x2, kernel_params):
    """
    Radial Basis Function (RBF) kernel.

    Parameters
    ----------
    x1, x2: array_like
        two d-dimensional vectors. 

    kernel_params: array_like
        a d-dimensional vector specifying the diagonals of a 
        covariance matrix which is inverted. 
    
    Returns: 
    ---------
    a scalar evaluating the RBF kernel at these two inputs 
    """
    h = (x1-x2).ravel()

    return jnp.exp(-jnp.sum(h**2 / kernel_params))

# Define the kernel function (e.g., RBF kernel)
def laplace_kernel(x1, x2, kernel_params):
    """
    Radial Basis Function (RBF) kernel.

    Parameters
    ----------
    x1, x2: array_like
        two d-dimensional vectors. 

    kernel_params: array_like
        a d-dimensional vector specifying the diagonals of a 
        covariance matrix which is inverted. 
    
    Returns: 
    ---------
    a scalar evaluating the RBF kernel at these two inputs 
    """
    h = (x1-x2).ravel()

    return jnp.exp(-jnp.sum(np.abs(h) / kernel_params))