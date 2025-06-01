# Load necessary packages
import math
import itertools
import numpy as np 
import scipy as scp
import matplotlib
import matplotlib.pyplot as plt

from scipy.linalg import eigh, expm 
from scipy.integrate import quad
from scipy import stats

###############################
# LEVEL SPACING DISTRIBUTIONS #
###############################

def poisson(s):
    return np.exp(-s)

def wigner_dyson(s):
    return (np.pi*s/2)*np.exp(-np.pi*(s**2)/4)

def wigner_dyson_complex(s):
    return 32*((s/np.pi)**2)*np.exp(-4/np.pi*(s**2))

def histogram_spectrum(matrix, WD_fit=False, poisson_fit=False, unitary=True, diagonalization=None, spacing_range=None, print_plot=True):
    """
    Plots the level spacing statistics (ratio) distributions.
    
    Parameters
    ----------
    
    matrix : array 
        D x D matrix of interest.
    
    WD_fit : bool, optional 
        Fits the Wigner-Dyson distribution
        
    poisson_fit : bool, optional
        Fits the Poisson distribution

    unitary: bool, optional
        Determines whether one compute eigenenergies or phases

    diagonalization: array, optional
        Eigensolution to the problem. Pair (eigvals, eigvecs) in numpy form

    spacing_range: tuple, optional
        Plot range

    print_plot: bool, optional
        Whether the plot is printed
                          
    Returns
    -------
    
    (spacing_dist, spacing_ratio) : (array, array)
        Distributions for the level spacing and the corresponding ratio distribution.      
    """ 

    # Passes the eigenvalues as an argument if available
    eigs = np.linalg.eig(matrix)[0] if diagonalization is None else diagonalization[0]

    # Complex angles if unitary
    if unitary:
        eigs = np.angle(eigs)

    # Spacing between the (ordered and consecutive) angles associated with the complex eigenvalues
    sorted_eigs = np.sort(eigs)
    spacing_dist = [abs(theta1 - theta0) for theta1, theta0 in zip(sorted_eigs, sorted_eigs[1:])]
    
    # Normalization. Important for comparing with the WD distribution
    spacing_dist = spacing_dist/np.mean(spacing_dist) 
    
    # Plot 
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(6, 3)
    fig.set_dpi(100)
    fig.suptitle(r'$\mathrm{Level \ spacing \ distribution}$')
    
    # Left plot
    
    # Range
    if spacing_range is not None:
         x_min, x_max = spacing_range
    else:
        x_min, x_max = np.min(spacing_dist), np.max(spacing_dist)
        
    # Plot and labels
    ax1.hist(spacing_dist, density=True, range=spacing_range, bins=30)  
    ax1.set_ylabel(r'$\mathrm{Counts}$')
    ax1.set_xlabel(r'$\mathrm{Spacing}$ $s_i = \theta_{i+1} - \theta_i$')
    
    # Level spacing ratio distribution
    spacing_ratio = [min(s1, s0)/max(s1, s0) for s0, s1 in zip(spacing_dist, spacing_dist[1:])]

    # Right Plot
    
    ax2.hist(spacing_ratio, density=True, bins=30)
    ax2.set_ylabel(r'$\mathrm{Probability}$')
    ax2.set_xlabel(r'$\mathrm{Data}$');
    
    # Wigner-Dyson fit
    if WD_fit:
        x = np.linspace(0, 4, 100)
        dist_fit_wdc = wigner_dyson_complex(x)
        ax1.plot(x, dist_fit_wdc, 'r')
    
    # Poisson distribution fit
    if poisson_fit:
        x = np.linspace(0, 4, 100)
        dist_fit_poss = poisson(x)
        ax1.plot(x, dist_fit_poss, '--', color='black')

    # Minimizes/removes the overlap between the plots
    fig.tight_layout()
    
    ax1.minorticks_on()
    ax2.minorticks_on()

    if print_plot:
        plt.show()
        print("Mean: ", np.mean(spacing_ratio))

    plt.close()
    
    return np.mean(spacing_dist), np.mean(spacing_ratio)