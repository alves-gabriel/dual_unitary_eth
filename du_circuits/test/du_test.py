import numpy as np
from ..src import dual_unitary as du

class TestCircuit:
    def test_permutator(self):
        
        pi_m = du.permutator([1, 2, 3, 4, 5, 6, 7])
        pi_m_expected = np.array([2, 4, 1, 6, 3, 7, 5])
        
        assert (pi_m_expected == pi_m).all(), "Incorrect permutation operator"
        
    def test_correspondent_statates(self):
        
        state_B = [1, 1, 0, 0, 0]
        state_A = [1, 0, 1, 0, 0]
        
        assert du.correspondent_statates(state_A, state_B)
    
    def test_soliton_action(self):
     
        a = np.array([1, 1, 1])
        m = np.array([1, 1, 0])
    
        assert du.soliton_action(a, m) == -1
    
    def test_charge_action(self):
        
        a = np.array([1, 0, 1, 0, 0])
        m = np.array([1, 0, 0, 0, 1])
        
        assert du.charge_action(a, m) == -3, "Incorrect quantum number"
    
    def test_scattering_matrix(self):
    
        L = 7
        V = du.scattering_matrix(L)
        expect_V = np.array([[0, 1, 0, 1, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 1, 0, 1, 0], 
                             [0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 1, 1], 
                             [0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0]])
    
        assert (V == expect_V).all()
    
    def test_cumulative_scattering(self):
        phi_s = du.cumulative_scattering([1, 1, 0, 0, 0])
        expected_phi_s = np.array([0, 1, 1, 2, 2, 2])
    
        assert (phi_s == expected_phi_s).all(), "Incorrect (cumulative) partial phase scattering list"

    def test_all_representatives_FKM(self):
    
        # For L = 5 and k = 2 (binary strings)
        L = 5
        rep_states =  np.array([[0, 1, 0, 0, 0], [0, 1, 0, 1, 0], [0, 1, 0, 0, 1], [0, 1, 0, 1, 1], [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]])
    
        assert (np.array(du.all_representatives_FKM(L)) == rep_states).all()

class TestNecklaces:
    def test_get_representatives(self):
    
        # Checks if we get (L m)/L representatives
        assert len(du.get_representatives(4, 11)) == 30, "Incorrect number of representative states for L = 11 and m = 4"
        
        # For L = 5 and m = 2. Check if states themselves are correct
        representatives = np.array([[0, 1, 0, 1, 0], [0, 1, 0, 0, 1]])
        
        assert (du.get_representatives(2, 5) == representatives).all(), "Incorrect representative states. An equivalent representation might have been used."
    
    
    def test_generate_necklaces(self):
        
        representatives = [[0, 0, 0, 0], 
                           [0, 0, 0, 1], 
                           [0, 0, 1, 1], 
                           [0, 1, 0, 1], 
                           [0, 1, 1, 1], 
                           [1, 1, 1, 1]]
        
        assert list(du.generate_necklaces(2, 4)) == representatives
        
    def test_sector_crossing(self):
    
        sector_representation = [1, 1, 0, 0, 0]
        
        assert du.sector_crossing(sector_representation, delete_diagonal=True) == 0.3584
        assert du.sector_crossing(sector_representation, delete_diagonal=False) == 0.36, "Incorrect crossing. Possibly wrong diagonal contributions?"