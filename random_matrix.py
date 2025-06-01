import numpy as np
import scipy as scp

def gamma_dist(k, lamb=1., size=None): 
    """ 
    See: https://github.com/scipy/scipy/blob/v1.13.0/scipy/stats/_continuous_distns.py#L3288-L3493
    for the scipy parametrization. 
    
    We use the wikipedia parametrization: https://en.wikipedia.org/wiki/Gamma_distribution
    """
    
    if size is None:
        return scp.stats.gamma.rvs(k, scale = 1/lamb) 
    else:
        return scp.stats.gamma.rvs(k, scale = 1/lamb, size=size) 

def goe_matrix(N, mu=0., sigma=1.):
    """Generates a N x N random matrix with normally distributed elements with mean mu and variance sigma."""
    
    mat_normal = np.random.normal(mu, sigma, (N, N))
    mat_normal = (mat_normal + mat_normal.conj().T)/np.sqrt(2)

    return mat_normal

def log_to_normal(mu_log, sigma_log):
    """Finds the value of mu and sigma such that the lognormal distribution X ~ exp(mu + sigma*Z), with Z ~ Normal(0, 1), 
    has the desired mean mu_log and variance sigma_log"""
    
    mu = np.log(mu_log**2/np.sqrt(mu_log**2 + sigma_log**2))
    sigma = np.sqrt(np.log(1 + sigma_log**2/mu_log**2))
    
    return mu, sigma

def lognormal_matrix(N, mu=1., sigma=1.):
    """Generates a N x N random matrix with lognormally distributed elements. 
    The underlying distribution has mean mu and variance sigma"""
    
    # Choose parameters so that the lognormal has same variance and mean as the gaussian
    mu_normal, sigma_normal = log_to_normal(mu, sigma)
    mat = np.random.lognormal(mu_normal, sigma_normal, (N, N))
    mat = (mat + mat.conj().T)/np.sqrt(2)

    return mat

def bernoulli_matrix(N, p=0.5):
    """Generates a N x N random matrix with lognormally distributed elements. 
    The underlying distribution has mean mu and variance sigma"""
    
    mat = np.random.binomial(n=1, p=p, size=(N, N))
    mat = (mat + mat.conj().T)/np.sqrt(2)
    
    return mat

def gamma_dist_matrix(N, k, lamb=1):
    """Generates a N x N random matrix with lognormally distributed elements. 
    The underlying distribution has mean mu and variance sigma"""
    
    mat = gamma_dist(k, lamb, size=(N, N))
    mat = (mat + mat.conj().T)/np.sqrt(2)
    
    return mat