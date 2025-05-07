from tqdm import tqdm
import jax
import jax.numpy as jnp
from jax import vmap
from copy import copy
from sklearn.preprocessing import StandardScaler
from .kernellibrary import *

# Gaussian Process Regression Class 
def K(X1, X2, kernel_func, kernel_params):
    '''
    Function for computing a kernel matrix

    Parameters
    ----------
    X1: array_like
        a dxN1 array of inputs where each column is an
        observation of a specific input. 
    X2: array_like
        a dxN2 array of inputs where each column is an
        observation of a specific input. 
    kernel_func: function
        a kernel function to apply element-wise to each
        entry of the kernel matrix
    kernel_params: array_like
        jax.numpy array of kernel parameters to send into
        the kernel function provided. 

    Returns: 
    -----------
    array_like: the N1 x N2 kernel matrix with the kernel
    function evaluated at each entry ij. 
    '''
    return vmap(
        vmap(lambda x, y: kernel_func(x,y, kernel_params), in_axes=(None,1)),
        in_axes=(1,None)
    )(X1, X2)

# Objective function for GPs (derived from MLE of Prior) 
def log_likelihood(p, kernel_func, X, Y, noise_var):
    """
    A function computing the loss for a specific set of 
    training data and kernel function. 

    Parameters
    ----------
    p: dict
        a dictionary containing the parameters for which we 
        are optimizing. In this case it's just 'kernel_params'
    
    kernel_func: function
        a function for which to form the kernel matrix. 

    X: array_like 
        a dxN array of input training data 

    Y: an Nx1 array of corresponding output training data 

    noise_var: the variance of any gaussian white noise in Y 

    Returns
    ----------

    The scalar loss-function value of the Marginal Likelihood 
    of the training data. 
    """

    Ktrain = K(X, X, kernel_func, p['kernel_params']) + noise_var * jnp.eye(X.shape[1])
    # Compute the scalar log-likelihood
    L = jnp.linalg.cholesky(Ktrain)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    quadratic_term = Y.T @ jax.scipy.linalg.cho_solve((L, True), Y) # Solve for (f.T Ktrain^-1 f)

    # Combine terms into a scalar
    loss = 0.5*(quadratic_term + logdet)

    return loss.squeeze()  # Ensure the input is a scalar

def adam_step(params, grads, lr, t, m, v, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Perform a single optimization step using the ADAM algorithm.

    Parameters:
        params (np.ndarray): Current parameter values.
        grads (np.ndarray): Gradient of loss with respect to parameters.
        lr (float): Learning rate.
        t (int): Current iteration count.
        m (np.ndarray): First moment vector (mean of gradients).
        v (np.ndarray): Second moment vector (mean of squared gradients).
        beta1 (float): Decay rate for the first moment.
        beta2 (float): Decay rate for the second moment.
        epsilon (float): Small constant for numerical stability.

    Returns:
        updated_params (np.ndarray): Updated parameters after the ADAM step.
        m (np.ndarray): Updated first moment vector.
        v (np.ndarray): Updated second moment vector.
    """

    # Update biased first and second moment estimates
    m = beta1 * m + (1 - beta1) * grads
    v = beta2 * v + (1 - beta2) * (grads ** 2)

    # Compute bias-corrected first and second moments
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Update parameters
    updated_params = params - lr * m_hat / (jnp.sqrt(v_hat) + epsilon)

    return updated_params, m, v

class GaussianProcess:
    """
    The main class for training, storing, and 
    optimizing Gaussian Processes. 
    """
    def __init__(self, kernel_func = rbf, double_precision = False, auto_scale = True):
        """
        The constructor of the Gaussian Process class

        Parameters
        ----------

        kernel_func: array_like (default = rbf)
            a kernel function if the user would like to pass it in. 

        double_precision: bool (default = False)
            specifying whether the user would like to use double
            precision floating point arithmetic (default is 32-bit). 

        rcond: float (default = 1e-10) many of the algorithms require linear solves and for 
            numerical stability it's often desirable to regularize results 
            by cutting off relative singular values below a certain tolerance. 

        auto_scale: bool (default = True)
            specifying whether the user would like to automatically 
            standard-scale the input and output data (recommended for parameter optimization)
        """
        # Set the kernel function 
        self.kernel_func = kernel_func

        # Enable 64-bit floating point precision 
        if double_precision:
            self.double_precision = True
            jax.config.update("jax_enable_x64", True)
        else:
            self.double_precision = False

        self.auto_scale = auto_scale


    def fit(self, X, Y, kernel_params, optimize_params = False, noise_var=1e-8, lr=1e-2, max_iter = 10000, max_stagnation = 100, verbose=True):
        """
        This is the function which trains the Gaussian Process model on its 
        training data. It does not optimize its hyperparameters. 

        Parameters
        ----------
        X: array_like
            The N x d array of input training data where each column represents 
            an observation of the input and d is the dimension of the 
            input. 

        Y: array_like
            the N x 1 array of output training data where the jth row represents 
            the output corresponding to the jth column of X. 

        kernel_params: array_like
            the parameters to be passed into the kernel function.

        noise_var: float (default = 1e-8)
            the variance of any Gaussian White Noise in Y 

        lr: float (default = 1e-2) 
            learning rate of the iterative algorithm 

        max_iter: int (default = 10000)
            maximum number of iterations of ADAM gradient descent steps 
            
        max_stagnation: int (default = 100)
            maximum number of steps without improvement in the 
            log-likelihood function. 

        verbose: bool (default = True)
            whether or not to print the results
        """
        # Kernel_params is a 1d numpy array containing kernel parameters 
        self.kernel_params = jnp.array(kernel_params)

        # noise_var is the variance of random-noise in Y
        # Pass in X (d x N) 2d array and Y, (N) 1d vectors
        self.X = jnp.array(jnp.copy(X).T) 
        self.Y = jnp.array(jnp.copy(Y).ravel())
        self.noise_var = noise_var

        if self.auto_scale:
            # Scaling X data 
            self.Xscaler = StandardScaler()
            self.X = self.Xscaler.fit_transform(self.X.T).T

            # Scaling Y data (but storing for later use)
            self.Ymean = jnp.mean(self.Y)
            self.Ystd = jnp.std(self.Y)
            self.Y = (self.Y - self.Ymean) / self.Ystd

        # Calling the hyperparameter optimization function
        if optimize_params:
            self.optimize_kernel_params(kernel_params, lr = lr, max_iter = max_iter, max_stagnation = max_stagnation, verbose = verbose)

        # Compute the training matrix 
        self.Ktrain = K(self.X, self.X, self.kernel_func, self.kernel_params) + noise_var * jnp.eye(self.X.shape[1])

        # Compute and store the cholesky decomposition of the training matrix
        self.L = jnp.linalg.cholesky(self.Ktrain)

        cond_num = jnp.linalg.cond(self.Ktrain)
        # Check condition number of kernel matrix 
        if cond_num > 1e8:
            print("Warning! Kernel Matrix is close to singular: K=%d" % (int(cond_num)))

        # Compute weights by solving linear system

        self.alpha = jax.scipy.linalg.cho_solve((self.L, True), self.Y)

    
    def predict(self, Xtest, include_std=True):
        """
        This function is for the online prediction of the training 
        data. 

        Parameters
        ----------
        Xtest: array_like
            The M x d array of testing inputs for which we would like 
            to approximate the value of the Gaussian Process for M 
            inputs. 

        include_std: bool (default = True)
            Whether or not to include the analytical standard deviation 
            associated with the Gaussian Process predictions 

        Returns
        ----------
        Yhat: A length-M array of model predictions at the inputs 

        Ystd: A length-M array of the standard deviation associated 
            with each prediction. 
        """
        # Scaling Xtest if necessary
        if self.auto_scale:
            Xtest = self.Xscaler.transform(Xtest)

        # Compute testing matrix 
        Ktest = K(Xtest.T, self.X, self.kernel_func, self.kernel_params) 

        # Expected value of test inputs 
        Yhat = Ktest @ self.alpha

        # Standard deviation of prediction at test inputs
        if include_std:
            # Compute the standard deviation of predictions
            Ystd = jnp.sqrt(jnp.diag(K(Xtest.T, Xtest.T, self.kernel_func, self.kernel_params) - Ktest @ jax.scipy.linalg.cho_solve((self.L, True), Ktest.T)))

            # Auto-scaling the predicted values if necessary
            if self.auto_scale:
                Yhat, Ystd = Yhat*self.Ystd + self.Ymean, Ystd * self.Ystd
            
            return Yhat, Ystd 
        else:
            # Auto-scaling the predicted values if necessary
            if self.auto_scale:
                Yhat = Yhat*self.Ystd + self.Ymean

            
            return Yhat
        
    def optimize_kernel_params(self, kernel_param_guess, lr=1e-2, max_iter = 10000, max_stagnation = 100, verbose=True):
        """
        This function is for optimizing the hyperparameters of the 
        Gaussian Process. This step is not required, but is necessary
        for the generalizability and accuracy of the model. 

        Parameters
        ----------
        kernel_param_guess: array_like
            the initial guess at the kernel parameters. 

        lr: float (default = 1e-2)
            The learning rate of the algorithm. 

        tol: float (default = 1e-6)
            The stopping tolerance of the algorithm. 
        
        max_iter: int (default = 10000)
            The maximum number of iterations of the optimization
            before force interrupting. 

        max_stagnation: int (default = 100)
            The maximum number of iterations without an improvement in the loss
            function before interrupting. 

        verbose: bool (default = True)
            Whether or not to print out the progress of the optimization. 
        """
        # Defining a parameter dictionary
        p = {
            'kernel_params':kernel_param_guess
        }

        grad_func = jax.grad(lambda p: log_likelihood(p, self.kernel_func, self.X, self.Y, self.noise_var))
        
        # Compute the gradient function of our loss-function
        initial_loss = log_likelihood(p, self.kernel_func, self.X, self.Y, self.noise_var)

        # Print the loss at the parameter guess 
        if verbose:
            print(f"Initial Loss: {initial_loss:.5f}")

        # Defining an iterator
        if verbose:
            iterator = tqdm(range(max_iter))
        else:
            iterator = range(max_iter)

        # Initializing best_loss variable
        best_loss = 1e99

        # Initializing moment vectors 
        m = jnp.zeros_like(p['kernel_params'])
        v = jnp.zeros_like(p['kernel_params'])

        # Looping through the iterator 
        for t in iterator:
            # Printing the loss at the step
            this_loss = log_likelihood(p,self.kernel_func, self.X, self.Y, self.noise_var)

            # Checking stagnation 
            if this_loss < best_loss:
                best_loss = this_loss
                best_params = copy(p)
                stagnation_count = 0 
            else:
                stagnation_count += 1
            
            # Break if we have not improved in 100 steps
            if stagnation_count > 100:
                print("No Improvements Made! Breaking Loop...")
                break

            # Break if the learning rate is zero to working precision
            if (lr < 3e-16 and self.double_precision) or (lr < 1e-7 and not self.double_precision):
                print("Learning-Rate is at machine precision...")
                break
            
            if verbose:
                iterator.set_postfix_str("Current Loss: %.5f Learning Rate: %.2e"  % (this_loss, lr))
            
            # Making a trial step
            grads = grad_func(p)

            trial_params, trial_m, trial_v = adam_step(p['kernel_params'], grads['kernel_params'], lr, t+1, m, v)
            
            
            trial_p = {
                'kernel_params':trial_params
            }

            # Taking the gradient at the trial step
            trial_grads = grad_func(trial_p)
            trial_loss = log_likelihood(trial_p,self.kernel_func, self.X, self.Y, self.noise_var)

            # Waiting until the trial step is valid i.e. no NaNs 
            while (jnp.isnan(trial_loss).any() or jnp.isnan(trial_grads['kernel_params']).any()):
                # Dividing the learning rate in half
                lr *= 0.5

                # Making a trial step 
                trial_params, trial_m, trial_v = adam_step(p['kernel_params'], grads['kernel_params'], lr, t+1, m, v)
                trial_p = {
                    'kernel_params':trial_params
                }

                # Taking the gradient at the trial step
                trial_grads = grad_func(trial_p)
                trial_loss = log_likelihood(trial_p,self.kernel_func, self.X, self.Y, self.noise_var)

            # Saving the next parameter step as the trial p 
            p, m, v = copy(trial_p), trial_m, trial_v

            
            
        # Save the best parameters
        p = best_params
        
        # Print final loss
        if verbose:
            print(f"Final Loss: {log_likelihood(p, self.kernel_func, self.X, self.Y, self.noise_var):.5f}\n")

        # Save best kernel params 
        self.kernel_params = p['kernel_params']






    
    
    

