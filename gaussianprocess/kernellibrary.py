import jax.numpy as jnp 

# Define the kernel function (e.g., RBF kernel)
def rbf(x1, x2, kernel_params):
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
    assert x1.shape[0] == x2.shape[0], "Input vectors are different dimensions!"
    assert len(kernel_params) == x1.shape[0], "Kernel parameters have different dimensions than inputs!"

    h = (x1-x2).ravel()

    return jnp.exp(-jnp.sum(h**2 / kernel_params))

def rbf_scaled(x1, x2,kernel_params):
    """
    Radial Basis Function (RBF) kernel with scaling. 

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
    assert x1.shape[0] == x2.shape[0], "Input vectors are different dimensions!"
    assert len(kernel_params) == x1.shape[0], "Kernel parameters have different dimensions than inputs! (dim(kernel_params) should equal dim(inputs)+1)"

    h = (x1-x2).ravel()

    return jnp.exp(-jnp.sum(h**2 / kernel_params[:h.shape[0]]))


def laplace_kernel(x1, x2, kernel_params):
    """
    Laplace kernel.

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
    assert x1.shape[0] == x2.shape[0], "Input vectors are different dimensions!"

    h = (x1-x2).ravel()

    return jnp.exp(-jnp.sum(jnp.abs(h) / kernel_params))

def nargp_kernel(x1, x2, kernel_params):
    """
    Kernel used in the Perdikaris + et al. Nonlinear Autoregressive GPs paper

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
    assert x1.shape[0] == x2.shape[0], "Input vectors are different dimensions!"

    y1 = x1[:-1]
    y2 = x2[:-1]
    return rbf(x1[-1], x2[-1], kernel_params[-1]) * rbf(y1, y2, kernel_params[:len(y1)]) + rbf(y1, y2, kernel_params[len(y1):2*len(y1)])
