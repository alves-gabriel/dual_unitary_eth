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
from functools import reduce
from functools import partial as partial_fun
import operator

from quantum_many_body.pauli_matrices import *

###################
# GATES AND UTILS #
###################

"""
This diagram illustrates the index change when we perform a tensor product:

 a      b         a  b
 |      |         |  |
 |      |         |  |
 c      d         c  d

I_{ac} I_{bd} -> I_{abcd}

'ac'   'bd'   -> 'abcd'
"""

# Kronecker product using einsum
def ein_kron(A, B):
    return np.einsum('ij,kl->ikjl', A, B)

# Reshape kronecker product, in matrix dimensions. The dimensionality of these functions do not coincide
def ein_kron_reshape(A, B):
    return np.einsum('ij,kl->ikjl', A, B).reshape(np.shape(A)[0]*np.shape(B)[0],np.shape(A)[1]*np.shape(B)[1])

# Dagger in tensor shape
def dagger(op):
    q = op.shape[0] # Single-site dimension
    return np.conj(op.reshape(q**2,q**2).T).reshape(q,q,q,q)

# CNOT gate (tensor form)
def CNOT():
    return ein_kron(np.outer(zero,zero), Id)+ein_kron(np.outer(one,one), X)

# Random GUE matrix
def GUE_random(N):
    H_GUE = np.random.normal(0, 1, size=(N, N))+1j*np.random.normal(0, 1, size=(N, N))
    H_GUE = (H_GUE + H_GUE.conj().T)/np.sqrt(2)

    return H_GUE

# See: Maris Ozols, How to generate a random unitary matrix (2009)
def haar_random(n):
    """Generates an N x N unitary matrix using the QR decomposition algorithm"""
    
    A, B = np.random.normal(0, 1, size=(n, n)), np.random.normal(0, 1, size=(n, n)) # 1. Sample matrices A and B (size n x n) from a normal distribution with mean 0 and variance 1.
    Q, R = np.linalg.qr(A + 1j*B)                                                   # 2. Calculate the QR decomposition of A + i B
    Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(n)])                 # 3. Calculate $\Lambda = diag(R_{ii}/|R_{ii}|)$
    return Q@Lambda                                                                 # 4. The matrix Q.Lambda is Haar random

# See: https://github.com/PieterWClaeys/UnitaryCircuits/blob/master/MaximumVelocityQuantumCircuits.ipynb
def dual_unitary(J, u_tuples=None, phi=None):

    # Parametrization of entangling two-qubit gate
    M = np.pi/4*(ein_kron_reshape(X, X)+ein_kron_reshape(Y, Y)+J*ein_kron_reshape(Z, Z))
    V = expm(-1j*M).reshape([2,2,2,2])
    
    # Random SU(2) unitaries
    if u_tuples is None:
        u_plus, u_min, v_plus, v_min = [haar_random(2) for _ in range(4)]
    else:
        u_plus, u_min, v_plus, v_min = u_tuples

    # Full parametrization of the dual-unitary circuit
    if phi is None:
        phi = np.pi*np.random.rand()
        
    U = np.exp(-1j*phi) * np.einsum('ac,bd,cdef,eg,fh-> abgh' ,u_plus, u_min, V, v_plus, v_min)
    
    return(U)

# Alternative form for the dual unitary
def du_perturbation(epsilon, J=0.5, tau=np.pi/4, H1=None, H2=None, H3=None, H4=None):

    # Parametrization of entangling two-qubit gate
    M = tau*(ein_kron_reshape(X, X)+ein_kron_reshape(Y, Y)+J*ein_kron_reshape(Z, Z))
    V = expm(-1j*M).reshape([2,2,2,2])

    if H1 is None:
        H1 = GUE_random(2)
    if H2 is None:
        H2 = H1
    if H3 is None:
        H3 = H1
    if H4 is None:
        H4 = H1
        
    UH1 = expm(-1j*epsilon*H1)
    UH2 = expm(-1j*epsilon*H2)
    UH3 = expm(-1j*epsilon*H3)
    UH4 = expm(-1j*epsilon*H4)
        
    v_plus, v_min, u_plus, u_min = UH1, UH2, UH3, UH4

    # Full parametrization
    U = np.einsum('ac,bd,cdef,eg,fh-> abgh', u_plus, u_min, V, v_plus, v_min)

    return U

def random_du():
    return dual_unitary(np.random.rand())

# Fixed Haar Random gates
Uhaar1=np.array([
    [0.67931743-1j*0.0721112, 0.51877751-1j*0.04466678, 0.25741725+1j*0.08032817, -0.36793521-1j*0.23261557],
    [0.3619221-1j*0.46028992, -0.56476001+1j*0.43994397, -0.09219379+1j*0.36114353, -0.07557832-1j*0.00214075],
    [0.13342533+1j*0.37674694, -0.30800024+1j*0.1579134j, 0.62386475-1j*0.20799404, -0.20374812+1j*0.49646409],
    [-0.13483809-1j*0.11203347, -0.30258211+1j*0.07080587, 0.25709566-1j*0.53832377, -0.15624451-1j*0.70170847]
    ])

Uhaar1, _ = scp.linalg.polar(Uhaar1)
Uhaar1 = Uhaar1.reshape(2,2,2,2)

Uhaar2=np.array([
    [-0.32936937+1j*0.0575935, -0.32742441+1j*0.30978811, -0.41818243+1j*0.42081727, -0.14116509-1j*0.55958207],
    [-0.53553904-1j*0.54762144, 0.13993909-1j*0.01096924, 0.5221732+1j*0.23437906, 0.23558662-1j*0.10249863],
    [-0.09194887+1j*0.40213391, -0.35024609+1j*0.37068525, 0.14797073+1j*0.19071808, 0.60159551+1j*0.38674017],
    [0.02435271-1j*0.36159117, -0.33621261+1j*0.63561192, 0.11961071-1j*0.49785786, -0.2804114+1j*0.10400783]
])

Uhaar2, _ = scp.linalg.polar(Uhaar2)
Uhaar2 = Uhaar2.reshape(2,2,2,2)

########################
# CIRCUIT ARCHITECTURE #
########################

# Tensor product 
def tensor_prod_simple(U, V):
    legs_U, legs_V = len(U.shape), len(V.shape)                                # Number of legs in each tensor
    index_U, index_V = list(range(legs_U)), list(range(legs_U, legs_U+legs_V)) # Appropriate indexing of their legs
        
        
    return np.einsum(U, index_U, V, index_V, 
                     index_U[:len(index_U)//2]+index_V[:len(index_V)//2]  # Upper legs
                     +index_U[len(index_U)//2:]+index_V[len(index_V)//2:] # Lower legs
                     , optimize='optimal'
                    )

# Rotates a list left
def rotate(l, shift):
    return l[shift:] + l[:shift]

# Converts a tensor product to a periodic configuration. Rotates left by default. Negative shift rotates right.
def periodicU(U, shift=1):
    sites=len(np.shape(U))
    index_upper=list(range(sites//2))     
    index_bottom=list(range(sites//2, sites)) 
    
    return np.einsum(U, index_upper+index_bottom, rotate(index_upper, shift)+rotate(index_bottom, shift), optimize='optimal')

# Parity transformation
def parity_reflection(U):
    sites=len(np.shape(U))
    index_upper=list(range(sites//2))     
    index_bottom=list(range(sites//2, sites))
    
    return np.einsum(U, index_upper+index_bottom, index_upper[::-1] + index_bottom[::-1] , optimize='optimal')

# Many-body tensor product (w/ sites). Receives a list of operators and corresponding sites 
# in op_site_list=[(op_A, site_A), (op_B, site_B), ...]. Applies the identities in places
# where there no operator is being applied.
# Example usage: tensor_prod([(X, 0), (Z, 1)], 3), tensor_prod([(Z, 0), (CNOT(), 1)], 3)
def tensor_prod(op_site_list, sites):
       
    """
        Gets a list of tuples [(op_A, site_A), (op_B, site_B), ...] from op_site_list. 
        Sorts it by site in increasing order. Operators form the array op_list, with the
        corresponding sites in site_list.
        
          |  |  |  |  |   |
          |  ____  |  __  |          
          | | U1 | | |U2| | ...                   
          | |____| | |__| |  
          |  |  |  |  |   |
          |  |  |  |  |   |
    """
    op_list, site_list = list(zip(*sorted(op_site_list, key=lambda op_site_pair:(op_site_pair[1], op_site_pair[0]))))

    tensor = op_list[0] if site_list[0]==0 else Id # Initializes the tensor 
    ind = len(np.shape(tensor))//2                 # Sweeps through the chain sites. Starts after the first operator.
       
    # Iteratively applies the unitaries
    while ind<sites:
        # Applies the operator corresponding to the current site
        if ind in site_list:
            U=op_list[site_list.index(ind)]
            tensor=tensor_prod_simple(tensor, U) 
            ind+=len(np.shape(U))//2
        # Applies the identity otherwise
        else:
            tensor=tensor_prod_simple(tensor, Id)
            ind+=1
    
    return tensor

# Returns a random matrix in the case of a completely random circuit.
# Otherwise, returns the same unitary everywhere
def SU4(U, random=False):
    return U if not random else haar_random(4).reshape(2, 2, 2, 2)

# Constructs even and wall layers of the brick-wall circuit
def _wall_construct(n_sites, parity, random=True, U=None, periodic=False):
    
    # Constructs layers of different parity
    if not periodic:
        if parity == 'even':
            UList=[(SU4(U, random), site) for site in list(range(0, n_sites-1 , 2))]+[(Id, n_sites)] #n_sites or n_sites-1 here?
        elif parity == 'odd':
            UList=[(Id, 0)]+[(SU4(U, random), site) for site in list(range(1, n_sites-1 , 2))]    
        else:
            raise ValueError('Parity is not even nor odd')
            
    # Takes care of the periodicity     
    elif periodic:
        UList=[(SU4(U, random), site) for site in list(range(0, n_sites-1 , 2))]
        
        if parity == 'odd': 
            return periodicU(tensor_prod(UList, n_sites))    
    
    return tensor_prod(UList, n_sites)

def brick_circuit(n_sites, T, random=True, U=None, U2=None, periodic=False):

    if U2 is None:
        U2=U
    
    # Number of sites depending on the periodicity
    if not periodic and n_sites % 2 == 0:
        raise ValueError('Open-ended chain should have an odd number of sites')
    elif periodic and n_sites % 2 == 1:
        raise ValueError('Periodic chain should have an even number of sites')

    # CAUTION: be sure this corresponds to the desired convention/correct order of multiplication
    index_bottom=list(range(0, 2*n_sites))                # Indices of lower legs
    index_upper=list(range(n_sites, 3*n_sites))           # Indices of upper legs
    output_index=list(set(index_upper)^set(index_bottom)) # Get the difference of the two sets using the ^ operator
    
    # Constructs the brick-wall circuit recursively, alternating between even and odd layers
    if T==0:
        return _wall_construct(n_sites, 'even', random, U, periodic)
    elif T>0:
        if T%2==1: 
            return np.einsum(_wall_construct(n_sites, 'odd', random, U2, periodic), index_bottom,
                             brick_circuit(n_sites, T-1, random, U, U2, periodic), index_upper,               
                             output_index, optimize='optimal')
        else: 
            return np.einsum(_wall_construct(n_sites, 'even', random, U, periodic), index_bottom,
                             brick_circuit(n_sites, T-1, random, U, U2, periodic), index_upper,
                             output_index, optimize='optimal')

#####################
# CIRCUIT OPERATORS #
#####################

def floquet_operator(L, U=Uhaar1, U2=None, periodic=False):    
    """Returns the Floquet operator U := U_2 U_1 for a circuit of size L."""

    if U2 is None:
        U2 = U
    
    return brick_circuit(L, 1, random=False, U=U, U2=U2, periodic=periodic).reshape(2**L, 2**L)

def extensive_pauli(L, op=Z, normalized=False):
    """Extensive pauli operator A := σ_0+σ_1...σ_L"""

    if normalized:
        norm = np.sqrt(np.ceil(L))
    else:
        norm = 1
        
    local_op_lst = np.array([op_local_circuit(L, op, site=l) for l in range(0, L)])
    
    return 1/norm * reduce(operator.add, local_op_lst)

def op_local_circuit(L, op=Z, site=None, normalized=False):
    """Local operator A := σ_{L/2}. Prefactor ensures that A.A=1 if 'normalized' flag is true."""

    if site==None:
        site=L//2
    
    operator_chain = np.array([np.eye(2)] * L, dtype = 'complex')
    operator_chain[site] = op
    
    if normalized:
        result_op = reduce(np.kron, operator_chain)
        norm = np.sqrt(np.trace(np.conjugate(result_op).T@result_op))
        return 1/norm*result_op
    else:
        return reduce(np.kron, operator_chain)

def op_extensive_circuit(L, op=Z, normalized=True):
    """Extensive operator A := σ_0+σ_2+...σ_L or σ_1+σ_3+...σ_L depending on the parity. Prefactor ensures that A.A=1"""

    if normalized:
        norm = np.sqrt(np.ceil(L/2))
    else:
        norm = 1
        
    local_op_lst = np.array([op_local_circuit(L, op, site=l) for l in range(0, L, 2)])
    
    return 1/norm * reduce(operator.add, local_op_lst)

def op_local_circuit_nearest(L, op=Z, site=None):
    """Local operator A := σ_{site} σ_{site+1}"""

    if site==None:
        site=L//2
    
    operator_chain = np.array([np.eye(2)] * L, dtype = 'complex')
    operator_chain[site] = op
    operator_chain[site + 1] = op
    
    return reduce(np.kron, operator_chain)

def op_extensiveNN(L, op=Z):
    """Circuit version of NN extensive operator"""

    local_op_lst = np.array([op_local_circuit_nearest(L, op, site=l) for l in range(0, L-1, 1)])
    
    return reduce(operator.add, local_op_lst)/np.sqrt(L//2)

def _kinectic_local_op(L, op, site, site_diff=1):
    """Local operator K_i := σ_{site} σ_{site+1}"""
    
    operator_chain = np.array([np.eye(2)] * L, dtype = 'complex')
    operator_chain[site] = op
    operator_chain[site + site_diff] = op
    
    return reduce(np.kron, operator_chain)

def op_NN_circuit_preserving(L):
    """Circuit version of the kinectic NN extensive operator. Normalized in the sense Tr(A^\\dagger.A)/D = 1"""

    local_op_lst = np.array([_kinectic_local_op(L, X, site=l, site_diff=1) for l in range(0, L-1, 1)]) \
                 + np.array([_kinectic_local_op(L, Y, site=l, site_diff=1) for l in range(0, L-1, 1)])
    local_op_lst = local_op_lst/np.sqrt(2*(L-1))
    
    return reduce(operator.add, local_op_lst)

def op_NNN_circuit_breaking(L):
    """Circuit version of the kinectic NN extensive operator. Normalized in the sense Tr(A^\\dagger.A)/D = 1"""

    local_op_lst = np.array([_kinectic_local_op(L, X, site=l, site_diff=2) for l in range(0, L-2, 1)]) \
                 + np.array([_kinectic_local_op(L, Y, site=l, site_diff=2) for l in range(0, L-2, 1)])
    local_op_lst = local_op_lst/np.sqrt(2*(L-2))
    
    return reduce(operator.add, local_op_lst)