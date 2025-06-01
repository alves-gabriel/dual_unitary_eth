import math
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

from .pauli_matrices import *

########################
# STATE REPRESENTATION #
########################

def get_int(array):
    """Convert a (classical) pure state to its decimal representation."""
    for i in range(len(array)):
        if array[-(i + 1)] == 1:
            return i

def spin_from_state(state):
    """Converts an interger to its binary representation, padding with zeroes. See: https://stackoverflow.com/a/68761692/17327385."""

    # Number of digits given by the dimension of Hilbert space
    num_bits = math.ceil(np.log2(len(state)))

    # Formats the string. ":0{num_bits}b" pads with the given number of zeroes
    return f'{get_int(state):0{num_bits}b}'

def state_from_spin(bitstring):
    """Converts bit string to a state."""

    # Defines up and down spin as vectors
    up = np.array([1, 0])
    down = np.array([0, 1])

    # Replaces bitstring with list of vectors
    spin_list = [up if bit == 1 else down for bit in bitstring]

    # Applies reduce in order to take the tensor product from everyone
    return reduce(np.kron, spin_list)

def spin_draw(binary, input_type = "binary", dtype = "int"):
    """
    Graphical representation of a binary string/spin state as a ket. For the latter, choose input_type='state". 
    Returns a ? for invalid entries in the list.
    """
    
    if input_type == "state":
        binary = spin_from_state(state)
        
    if dtype=="int":
        return '|' + ''.join(['⚫' if s == 1 else '⚪' if s == 0 else '?' for s in binary]) + '〉'
        
    elif dtype=="string":
        return '|' + ''.join(['⚫' if s == '1' else '⚪' if s == '0' else '?' for s in binary]) + '〉'

#########################
# EIGENSTATE PROPERTIES #
#########################

def same_state(v, w, tol=1e-1, return_differences=False):
    """
    Check if two vectors u and w are equal up to a global phase shift.

    Returns:
    
    - True if w is a global phase shift of v, False otherwise.
    - The difference vector (if return_differences is True)
    """    
    
    # Computes the phase factors for non-zero elements
    nonzero_indices = np.array(np.abs(v) > tol)
    ratios = w[nonzero_indices]/v[nonzero_indices]

    # Edge case: if v and w belong to entirely different subspaces the ratio might be 0 everywhere. Return False here
    if np.allclose(ratios, 0, atol=tol):
        return False, None
    
    # Computes the phases
    phases = np.angle(ratios)

    # Checks if all phases are equal, up to the tolerance
    if return_differences:
        return np.allclose(phases, phases[0], atol=tol), np.angle(w[nonzero_indices]) - np.angle(v[nonzero_indices])
    else:
        return np.allclose(phases, phases[0], atol=tol)

def is_eigenstate(v, U, tol=1e-1, return_eigenvalue=True):
    """
    Check if the vector v is an eigenstate of the unitary U

    Returns:
    
    - True if v is an eigenvalue of U
    - The corresponding eigenvalue lambda = (U.v)/v
    """    
    
    is_matching, angle = same_state(v, U@v, tol=tol, return_differences=True)
    eigenvalue = np.exp(1j * angle[0]) if is_matching else None

    return is_matching, eigenvalue

def plot_quasienergies(eigvals, radius=1, **kwargs):
    """Plot quasi-energies in a unit circle"""
    fig, ax = plt.subplots()
    
    # Unit circle
    t_lst = np.linspace(0, 2*np.pi, 100)
    ax.plot(radius * np.cos(t_lst), radius * np.sin(t_lst), "--", color="black")
    ax.set_aspect(1)

    # Plot eigenvalues
    ax.scatter(eigvals.real, eigvals.imag, **kwargs)
    ax.set_xlabel(r"Re $\lambda$")
    ax.set_ylabel(r"Im $\lambda$")

    return fig, ax

###################
# DIAGONALIZATION #
###################

def operator_block(A, sec_proj, eig_proj, m, n):
    """"
    Returns the block-operator between sectors m and n.
    
    Given the projection P_m, into the magnetization sector m and
    the unitary V_m build from the eigenbasis of the corresponding
    unitary sector, we have:
    
    A_{mn} = V_m P_m A P†_n V†_n
    """
    
    return eig_proj[m]@sec_proj[m]@A@sec_proj[n].conj().T@eig_proj[n].conj().T