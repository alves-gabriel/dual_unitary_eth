import numpy as np
import scipy as scp
import itertools
from functools import reduce

#########
# GATES #
#########

def dual_unitary_cphase(phi):
    """"Dual-unitary gate in SWAP + controlled phase form."""
    
    # SWAP gate
    swap = np.array([[1, 0, 0, 0], 
                     [0, 0, 1, 0], 
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])

    # Controlled phase
    c_phase = np.diag([np.exp(1j*phi), 1, 1, 1])

    # Returns DU gate as S.CP(phi)
    return c_phase@swap

################################
# BITSTRING CIRCUIT OPERATIONS #
################################

def permutator(permutation):
    """
    Implements the SWAP circuit with open-boundary conditions:
    
        - Right moving nodes are updated as n -> n + 2,
        - left moving nodes are updated as n -> n - 2,
        - and edges are updated as 0 -> 1 and L - 1 -> L - 2.

    Returns the permutation of a (integer) list.

    Usage example:
    >>> permutator([1, 2, 3, 4, 5, 6, 7])
    array([2, 4, 1, 6, 3, 7, 5])
    """
    
    # Initializes
    L = len(permutation)
    permutated = np.zeros(L, dtype="int")

    # Right moving
    for site in range(0, L - 2, 2):
        permutated[site + 2] = permutation[site]
        
    # Left moving
    for site in range(3, L - 1, 2):
        permutated[site - 2] = permutation[site]

    # Edges
    permutated[0] = permutation[1]
    permutated[L - 2] = permutation[L - 1]

    return permutated

def generate_all_permutations(bitstring):
    """Gets all permutations associated with a bitstring"""

    strings = []
    for _ in range(len(bitstring)):
        strings.append(bitstring)
        bitstring = permutator(bitstring)
                       
    return strings

def bitstring_to_soliton(bitstring):
    """Converts a bitstring to an operator"""

    # Pauli Gates
    Id = np.array([[1, 0], [0, 1]])
    Z = np.array([[1, 0], [0, -1]])
    
    operator_list = [Id if s == 0 else Z for s in bitstring]

    return reduce(np.kron, operator_list)

def generate_all_charges(L):
    """Generates all charges, in matrix form, for a circuit of size L"""
    
    charges = []
    
    for generating_soliton in all_representatives_FKM(L):
        charges.append(np.sum(list(map(bitstring_to_soliton, generate_all_permutations(generating_soliton))), axis=0))

    return charges

def correspondent_statates(state_A, state_B):
    """Checks whether there is a permutation π^s(...) such that π^s(state_B) = state_A
    
    Usage example:
    >>> correspondent_statates([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    True
    """
    
    if len(state_A) != len(state_B):
        raise ValueError('Objects of unequal length')

    permutated_state = state_A

    # Tries all the L permutations
    for _ in range(len(state_A)):

        # Found a matching permutation
        if permutated_state == state_B:
            return True

        # Permutes the state
        permutated_state = permutator(permutated_state).tolist()

    # No matching permutations found
    return False   

def soliton_action(soliton, state):
    """
    Returns the eigenvalue of the soliton q^(m)_(a), given the soliton (a) and the sate |m〉.
    
    Usage example:
    >>> a = np.array([1, 1, 1])
    >>> m = np.array([1, 1, 0])
    >>> soliton_action(a, m).item()
    -1
    """

    return (-1)**(np.sum(soliton) - np.sum(soliton*state))

def charge_action(soliton, state):
    """
    Returns the eigenvalue of the charge Q^(n)_(a), given the soliton (a) and the sate |m〉.

    Usage example:
    >>> a = np.array([1, 0, 1, 0, 0])
    >>> m = np.array([1, 0, 0, 0, 1])
    >>> charge_action(a, m).item()
    -3
    """
    
    # Checks if dimensions are the same
    if len(soliton) != len(state):
      raise ValueError("Dimensions do not match")

    # Initializes the quantum number
    eigenvalue = 0
    L = len(soliton)

    # Iterates over the l different solitons
    for l in range(L):

        # Contribution of each term
        eigenvalue += soliton_action(soliton, state)
        soliton = permutator(soliton)

    return eigenvalue

################################
# SCATTERING PHASE COMPUTATION #
################################

def scattering_matrix(L):
    """
    Defines the scattering matrix V for a L x L sector, which is used to compute the phase acquired by a given state.
    In this convention, we have V_ij - 1 if the i-th bit is controlled j-th one. For instance, we have, 
    
    V[0, 1] = V[0, 3] = 1, 
    
    because the bit on site zero is controlled by the bits in sites 1 and 3.
    """
    
    # Create an L x L sparse matrix
    matrix = scp.sparse.lil_matrix((L, L), dtype=int)

    # Boundary of the circuit 
    matrix[L - 3, L - 2] = 1  
    matrix[L - 3, L - 1] = 1  

    # Fills entries
    for i in range(L - 3):
        # Even rows only
        if i % 2 == 0:  
            matrix[i, i + 1] = 1 # Control coming from site i + 1
            matrix[i, i + 3] = 1 # Control coming from site i + 3

    return matrix.toarray()

def scattering_phase(b):
    """
    Given the scattering matrix V, the (multiplicity of the) phase picked up by a state (b) is given by (b|V|b).

    Usage example:
    >>> scattering_phase([1, 1, 0, 1, 0]).item()
    2
    """
    
    L = len(b)    
    return b@scattering_matrix(L)@b

def phases_list(rep_state):
    """Returns a list with the multiplicity of the (partial) phase acquired by states in the sector defined by rep_state."""

    # Initialization
    phases = []
    state = rep_state

    # L iterations through sector
    for _ in range(len(rep_state)):
        phases.append(scattering_phase(state)) # Phase picked up by the state
        state = permutator(state)              # Permutes the state's bitstring
        
    return phases
    
def cumulative_scattering(rep_state):
    """A list with the partial scattering phase (multiplicity) [Phi_s] of the sector determined by rep_state.
    
    Usage example:
    >>> cumulative_scattering([1, 1, 0, 0, 0])
    array([0, 1, 1, 2, 2, 2])
    """

    # By definition, Phi_0 = 0
    Phi_0 = 0
    return np.cumsum([Phi_0] + phases_list(rep_state))

#############################
# ANALYTICAL EIGENSOLUTIONS #
#############################

def quasi_energies(phi, rep_state):
    """Given the phase phi, returns a list with the quasi-energies of the sector given by rep_state."""

    # Total scattering phase
    L = len(rep_state)
    Phi = phi * cumulative_scattering(rep_state)[L]

    # Momenta
    k = (2*np.pi*np.arange(0, L) - Phi)/L
    
    return np.exp(-1j*k)
    
def circuit_eigenstate(phi, n, rep_state):
    """Returns the n-th eigenstate of the sector given by rep_state and circuit phases phi"""

    # Total scattering 
    L = len(rep_state)
    Phi = phi * cumulative_scattering(rep_state)[L]

    # Partial scattering (as a list)
    Phi_s = phi * cumulative_scattering(rep_state)[:-1]

    # Momenta (as a list)
    k = (2*np.pi*np.arange(0, L) - Phi)/L

    # Constructs the eigenstate with ordering (|m>, |π(m)>, ...)
    eigenstate = 1/np.sqrt(L) * np.exp(1j*(k[n] * np.arange(0, L) + Phi_s)) 

    return eigenstate   

def sector_crossing(sector_representation, delete_diagonal=True, two_body_solitons=None):
    """
    Returns the crossing contribution from the sector (of size L) determined by sector_representation

    Usage example:
    >>> sector_crossing([1, 0, 1, 0, 0], delete_diagonal=True).item()
    0.3584
    """

    L = len(sector_representation)
    
    # Magnetization. Note that M takes values $ - L, -L + 1 , ..., -1, 0, 1, ..., L$ in this convention
    sector_representation = np.array(sector_representation)
    M =  2 * sum(sector_representation) - L
    
    # Charge of each representative two-body soliton
    lambda_lst = []

    # No pre-computed two-body solitons
    if two_body_solitons is None:
        two_body_solitons = get_two_body_solitons(L)

    # Loops over all the representative state of order 2
    for second_order_representation in get_two_body_solitons(L):
        lambda_lst.append(charge_action(second_order_representation, sector_representation))

    lambda_lst = np.array(lambda_lst)

    # Crossing diagram. Might include correction for diagonal terms or not
    if delete_diagonal:
        return 2*np.sum(lambda_lst**2)/L**3 + 1/L - M**4/L**4
    else:
        return 2*np.sum(lambda_lst**2)/L**3 + 1/L 

###############################
# CHARGES AND REPRESENTATIVES #
###############################
    
def nbody_charges(n, L):
    """Generates all the n-body solitons/states in a spin chain of sizes L. Yields a generator with each permutation."""

    # Find all the combinations of where the up spins can be placed
    for up_positions in itertools.combinations(range(L), n):        
        
        # Array filled with zeroes
        soliton = [0]*L

        # Up spins
        for i in up_positions:
            soliton[i] = 1

        # Generator
        yield soliton

def all_representatives_FKM(L, include_fully_pol = False):
    """Returns all the representative states with FKM algorithm

    Usage example:
    >>> all_representatives_FKM(5)
    [array([0, 1, 0, 0, 0]), array([0, 1, 0, 1, 0]), array([0, 1, 0, 0, 1]), array([0, 1, 0, 1, 1]), array([0, 1, 1, 1, 0]), array([0, 1, 1, 1, 1])]
    """

    # Uses the FKM algorithm for k = 5 (binary) necklaces
    representative_cycles = generate_necklaces(2, L)

    # Goes from cycle notation to bitstring representation of solitons
    representative_permutations = list(map(cycle_to_site, representative_cycles))

    # Might include fully polarized states or not
    return representative_permutations[1 : -1] if not include_fully_pol else representative_permutations

def get_representatives(m, L):
    """
    Returns all the representative states of magnetization m for strings of size L.

    Usage example:
    >>> get_representatives(2, 5)
    [array([0, 1, 0, 1, 0]), array([0, 1, 0, 0, 1])]
    >>> len(get_representatives(3, 7))
    5
    """
    
    representative_states_m = []

    # Runs over all m-body solitons, adding representative states to the list above
    for string in all_representatives_FKM(L):
        if np.sum(np.array(string)) == m:
            representative_states_m.append(string)

    return representative_states_m

def get_two_body_solitons(L):
    """Constructs all the (L-1)/2 two-body solitons.
    
    Usage example:
    >>> get_two_body_solitons(7)  
    [array([1, 1, 0, 0, 0, 0, 0]), array([1, 0, 0, 1, 0, 0, 0]), array([1, 0, 0, 0, 0, 1, 0])]
    """
    
    solitons = []

    # Fix the first soliton and site zero, w/ the other "jumping" every other site
    for i in range((L - 1)//2):
        soliton = np.zeros(L, dtype="int")
        soliton[0] = 1
        soliton[2 * i + 1] = 1
        solitons.append(soliton)

    return solitons

#####################################
# NECKLACE ALGORITHM IMPLEMENTATION #
#####################################

def generate_necklaces(k, n):
    """ An iterative algorithm to generate all the k-ary necklages of length n.

    References: 
        - [1] https://www.sciencedirect.com/science/article/pii/S0196677400911088?ref=cra_js_challenge&fr=RR-1
        - [2] https://www.sciencedirect.com/science/article/pii/019667749290047G

    Usage example:
    >>> list(generate_necklaces(2, 4))
    [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1], [1, 1, 1, 1]]
    """

    # Starting string. Note that a[0] = 0 
    a = [0] * (n + 1)

    # First necklaces (0, 0, ..., 0)
    yield a[1:]

    i = n
    while i > 0:
        a[i] = a[i] + 1
        
        for j in range(1, n - i + 1):
            a[j + i] = a[j]

        if n % i == 0:
            yield a[1:]

        i = n

        while a[i] == k - 1:
            i = i - 1

def site_to_cycle(array):
    """ Converts the permutation rules in the SWAP circuit to cycle notation. Namely, to

    [x, sigma(x), sigma(sigma(x)), ...]

    This means that in [s_0, s_1, ..., s_n] the soliton in site s_i goes to site s_{i+1}.

    For instance, for the ordered sites [0, 1, 2, 3, 4, 5, 6] we get [0, 2, 4, 6, 5, 3, 1],
    which means that (the soliton in) site 0 goes to site 2, site 2 goes to site 4 and so on.

    Usage example:
    >>> site_to_cycle([1, 2, 3, 4, 5, 6, 7])
    array([1, 3, 5, 7, 6, 4, 2])
    """
    
    # Initializes
    L = len(array)
    indices_set = np.zeros(L, dtype="int")

    # Right moving
    for j in range(0, L//2 + 1):
        indices_set[j] = 2*j
        
    # Left moving
    for j in range(1, L//2 + 1):
        indices_set[L//2 + j] = L - 2*j
    
    return np.array(array)[indices_set]

def cycle_to_site(array):
    """Applies a permutation with cycle notation.
    
    Usage example:
    >>> cycle_to_site([1, 3, 5, 7, 6, 4, 2])
    array([1, 2, 3, 4, 5, 6, 7])
    """
    
    # Initializes
    L = len(array)
    cycle_indice_set = site_to_cycle(range(L))
    sites = np.zeros(L, dtype="int")

    # Right moving
    for j in range(L):
        sites[cycle_indice_set[j]] = array[j]
        
    return sites