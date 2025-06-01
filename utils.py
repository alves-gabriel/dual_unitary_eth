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

# https://stackoverflow.com/questions/40441391/how-to-print-a-2d-array-in-a-pretty-format
def pretty_print(a, is_int=False):
    for row in a:
        for col in row:
            if is_int:
                print("{:2d}".format(col), end=" ")
            else:
                print("{:5.2f}".format(col), end=" ")
        print("")

# Hybrid log linear scale, forward and inverse scaling functions

def forward_log_lin(threshold, x_m, x_M, x, ratio=.5):
    """ Defines a scaling function which is logarithmic up to 'threshold' and linear after that. That is, a half log half linear scale.
    
    TODO: check if this works for threshold != 1"""

    # Manually deals with the point x = 0.
    # Furthermore, we have to subtract 'threshold' here to make the end and the beginning of each scale coincide (otherwise we get a gap)
    # For instance, if x = 1 we have log(x) = 0, at the left side, and x = 1 at the linear part, so we have no points between [0, 1] in the linear scale,
    # it appears to have a gap. Subtracing the threshold seems to solve it. Meanwhile, we have to ADD it in the inverse function.
    alpha = (threshold - x_M)*(1/ratio - 1)/(threshold*np.log10(x_m/threshold))
    
    x_transf = [((alpha*np.log10(x[i]/threshold) + 1)*threshold if x[i] < threshold and x[i] != 0 else x[i]) for i in range(len(x))]

    return np.array(x_transf)  

def inverse_log_lin(threshold, x_m, x_M, x, ratio=.5):
    """Inverse of forward_log_lin"""
    alpha = (threshold - x_M)*(1/ratio - 1)/(threshold*np.log10(x_m/threshold))
    
    y_transf = [(10**((x[i]/threshold-1)/alpha)*threshold if x[i] < threshold and x[i] !=0  else x[i]) for i in range(len(x))]
    
    return np.array(y_transf)  

def log_lin_space(x_i, x_middle, x_f, N_points_log, N_points_lin=None, endpoint=True):
    """"For a better resolution on the large frequency regime, we want a scale which is logarithimic up to w ~ 1 
    and linear afterwards, since we have w < 2*pi for circuits

    x_i, x_middle, x_f:
        Intervals [x_i, x_middle] in logscape and [x_middle, x_f] for linear scale
    N_points_log:
        Number of points in the log part
    endpoints:
        Whether to include endpoint
    """
    
    # See https://stackoverflow.com/questions/3305865/what-is-the-difference-between-log-and-symlog
    if N_points_lin is None:
        N_points_lin = N_points_log

    # Separation of scales. Logarithmic up to x_middle.
    log_points = np.logspace(np.log10(x_i), np.log10(x_middle), N_points_log, endpoint=False)
    lin_points = np.linspace(x_middle, x_f, N_points_lin, endpoint=endpoint)

    return np.concatenate((log_points, lin_points))

##########
# COLORS #
##########

# Sites to build palettes:
# https://www.schemecolor.com/blue-green-ocean-color-scheme.php

# Colors and Style
blue_colors = ['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#2b8cbe','#045a8d']
green_colors = ['#edf8fb','#ccece6','#99d8c9','#66c2a4','#2ca25f','#006d2c']
red_colors = ['#fee5d9','#fcbba1','#fc9272','#fb6a4a','#de2d26','#a50f15']
purple_colors = ['#f2f0f7','#dadaeb','#bcbddc','#9e9ac8','#756bb1','#54278f']
orange_colors = ['#feedde','#fdd0a2','#fdae6b','#fd8d3c','#e6550d','#a63603']

# Custom Diverging palettes
# Edit on: https://color.adobe.com/create/color-wheel
palette_orange_blue = np.array(["#FF6D00", "#FF8500", "#FF9E00", "#00B4D8", "#0077B6", "023E8A"])
palette_blue_lime = np.array(["#184E77", "#1A759F", "#34A0A4", "#76C893", "#B5E48C", "#D9ED92"])
palette_amaranth_cerulean = np.array(["#1780a1", "#455e89", "#5c4d7d", "#723c70", "#a01a58", "#b7094c"])
palette_ocean_blue_green = np.array(["#5982BD", "#91B8E6", "#CCE4DF", "#A6D6AD", "#8EBD6F"])

# From: https://colorbrewer2.org/#type=sequential&scheme=BuGn&n=6
marker_lst = ['*', 'o', 's', 'd', 'x', 'v', '^'] 