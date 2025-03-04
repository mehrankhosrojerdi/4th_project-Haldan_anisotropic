import quimb as qu
import quimb.tensor as qtn
import numpy as np
from joblib import Parallel, delayed


class Haldan_anis_unsupervised:

    #-----------------------------------------------------------------------#
    def __init__(self, L, ls):
        self.L = L 
        self.ls = ls 
    #-----------------------------------------------------------------------#
    def MPO(self, D, E):
        """Constructs the MPO for the anisotropic Haldane chain.
        
        Args:
            J (float): Heisenberg coupling strength.
            D (float): Single-ion anisotropy coefficient.
            E (float): In-plane anisotropy coefficient.

        Returns:
            qtn.MatrixProductOperator: MPO representation of the Hamiltonian.
            
        """
        J=1
        I = qu.eye(3).real  # Identity matrix for spin-1
        Sx = qu.spin_operator('X', S=1).real
        Sy = qu.spin_operator('Y', S=1).real
        Sz = qu.spin_operator('Z', S=1).real
        
        # Define the MPO tensor (5x5 bond dimension)
        W = np.zeros([5, 5, 3, 3], dtype=float)

        # Fill in the MPO tensor
        W[0, 0, :, :] = I  # Identity propagation
        W[0, 1, :, :] = J * Sx
        W[0, 2, :, :] = J * Sy
        W[0, 3, :, :] = J * Sz
        W[0, 4, :, :] = D * (Sz @ Sz) + E * (Sx @ Sx - Sy @ Sy)  # On-site terms

        # Interaction terms between sites
        W[1, 4, :, :] = Sx
        W[2, 4, :, :] = Sy
        W[3, 4, :, :] = Sz
        
        # Final identity propagation term
        W[4, 4, :, :] = I

        # Left and right boundary terms
        Wl = W[0, :, :, :]  # Left boundary tensor
        Wr = W[:, 4, :, :]  # Right boundary tensor

        # Build the MPO
        H = qtn.MatrixProductOperator([Wl] + [W] * (self.L - 2) + [Wr])

        return H
    #-----------------------------------------------------------------------#
    def DMRG(self, d1, e1):
        DMRG = qtn.tensor_dmrg.DMRG(ham = Haldan_anis_unsupervised(L = self.L, ls = self.ls).MPO(D = d1, E = e1), bond_dims = 30) 
        DMRG.solve(tol = 1e-8, verbosity = 0);
        ground_state = DMRG.state
        return ground_state
    #-----------------------------------------------------------------------#
    def generate_Entire_set(self):

        # Define ranges for E and D
        D = np.linspace(-2, 2, int(self.ls))
        E = np.linspace(-2, 2, int(self.ls))
        
        result = np.array(np.meshgrid(D, E)).T.reshape(-1, 2)
        lst_DMRG_state = Parallel(n_jobs=5, backend='loky')(delayed(Haldan_anis_unsupervised(L = self.L, ls = self.ls).DMRG)(d, e) for d, e in result)


        return np.array(lst_DMRG_state)