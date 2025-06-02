[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15576373.svg)](https://doi.org/10.5281/zenodo.15576373)


# Probes of Full Eigenstate Thermalization in Ergodicity-Breaking Quantum Circuits
---

Here we provide the code and a minimal example for the results in the paper. The files as divided as:

- `example.ipynb` : a Jupyter notebook with an minimal example
- `du_circuits` : a library implementing all circuit operations, including the analytical results for the diagonaliztion of the integrable DU XXZ circuit
    -  `src` : source code and documentation with examples
    -  `test` : unit tests with some further examples
    -  `example_du` : a short example
- `quantum_many_body` : a folder with some utilities for QMB operations and operators
- `free_proability.py` : a class for performing diagram computations and free probability operations
- `random_circuits.py` : code implementing random circuits of somewhat general configurations. This is inherited from previous code, so the functionalities here are much more general than what is required in the paper (TODO: this might be simplified in a feature commit).

### How to cite this
---


If you are interested in citing this, feel free to check the repository in Zenodo associated with the [DOI](https://zenodo.org/records/15576373) in the badge above.