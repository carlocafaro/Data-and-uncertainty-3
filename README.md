Data-and-uncertainty-3
======================
import numpy as np
import pylab as pl

"""
File containing code relating to Computational Lab 3 of the Dynamics
MPECDT core course.
"""

def identity(x):
    """
    Function to compute the square of the input. If the input is 
    a numpy array, this returns a numpy array of all of the squared values.
    inputs:
    x - a numpy array of values

    outputs:
    y - a numpy array containing the square of all of the values
    """
    return x

def normal(N):
    """
    Function to return a numpy array containing N samples from 
    a N(0,1) distribution.
    
    inputs:
    N - number of samples
    
    outputs:
    X - the samples
    """
    return np.random.randn(N)

def normal_pdf(x):
    """
    Function to evaluate the PDF for the normal distribution (the
    normalisation coefficient is ignored). If the input is a numpy
    array, this returns a numpy array with the PDF evaluated at each
    of the values in the array.

    inputs:
    x - a numpy array of input values
    
    outputs:
    rho - a numpy array of rho(x)
    """
    return np.exp(-x**2/2)

def uniform_pdf(x):
    """
    Function to evaluate the PDF for the standard uniform
    distribution. If the input is a numpy array, this returns a numpy
    array of all of the squared values.

    inputs:
    x - a numpy array of input values
    
    outputs:
    rho - a numpy array of rho(x)
    """
    y = np.ones(x.shape)
    y[x<0] = 0.0
    y[x>1] = 0.0
    return y

def importance(f, Y, rho, rhoprime, N):
    """
    Function to compute the importance sampling estimate of the
    expectation E[f(X)], with N samples

    inputs:
    f - a Python function that evaluates a chosen mathematical function on
    each entry in a numpy array
    Y - a Python function that takes N as input and returns
    independent individually distributed random samples from a chosen
    probability distribution
    rho - a Python function that evaluates the PDF for the desired distribution
    on each entry in a numpy array
    rhoprime - a Python function that evaluates the PDF for the
    distribution for Y, on each entry in a numpy arraynp.random.randn(N)
    N - the number of samples to use
    """
    Y=Y(N)
    theta = 1/(np.sum(rho(Y)/rhoprime(Y)))*np.sum(f(Y)*rho(Y)/rhoprime(Y))
   


    return theta
if __name__ == '__main__':
    
    
     Ns=np.array([10,100,1000,10000,100000,1000000,10000000])
     err=np.zeros(Ns.shape)

     for i,N in enumerate(Ns):
	 theta = importance(identity,normal,
                       uniform_pdf,normal_pdf,N)
         err[i]=np.abs(0.5 - theta)
         print err
     pl.clf()
     pl.loglog(Ns, err,   'k',   label='error')
     pl.show()
    
