import numpy as np

# Down and up spins
zero = np.array([0, 1])
one = np.array([1, 0])

# Pauli matrices
Id = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
