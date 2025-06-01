# Load necessary packages
import math
import numpy as np 
import scipy as scp
import matplotlib
import matplotlib.pyplot as plt

def _nest_list(f, x, applications):
    """Auxiliary function for nest_list -- does not keep whole list in memory"""
    for i in range(applications):
        yield x
        x = f(x)
        
    yield x

def nest_list(f, x, applications):
    """Nested list which behaves like its Mathematica counterpart"""
    
    return list(_nest_list(f, x, applications))

def chop(array, tol=1e-6):
    """Sets any elements smaller than tol to zero"""
    
    array = np.array(array)
    array[np.abs(array) < tol] = 0.0
    
    return array