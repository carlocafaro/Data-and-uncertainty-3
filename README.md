Data-and-uncertainty-3
======================
import numpy as np
import pylab as pl

"""
File containing code relating to Computational Lab 3 of the Dynamics
MPECDT core course.
"""

def newfunction(x):
    """
    Function to compute the square of the input. If the input is 
    a numpy array, this returns a numpy array of all of the squared values.
    inputs:
    x - a numpy array of values

    outputs:
    y - a numpy array containing the square of all of the values
    """
    return np.exp(-10.0*x)*np.cos(x)

def uniform(N):
    """
    Function to return a numpy array containing N samples from 
    a uniform distribution.
    
    inputs:
    N - number of samples
    
    outputs:
    X - the samples
    """
    return np.random.rand(N)



def sample_cluster(N):

    u=uniform(N)
    
    x=-0.1*np.log(1-u+u*np.exp(-10.0))

    return x

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


def cluster_pdf(x):
    
    "Function to evaluate the PDF for a function useful for the importance sampling that is more concentrated in zero than newfunction"
    y = np.zeros(x.shape)
    y = 10.0*np.exp(-10.0*x)/(1-np.exp(-10.0))
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
	 theta = importance(newfunction,sample_cluster,
                       uniform_pdf,cluster_pdf,N)
        
         err[i]=np.abs(10.0/101.0-(10.0*np.cos(1.0)-np.sin(1.0))/(101.0*np.exp(10.0)) - theta)
         "Comparing the analytic expected value with that produced by the importance sampling method increasing the number of samples N" 
         
     pl.clf()
     pl.loglog(Ns, err,   'k',   label='error')
     pl.show()
    
