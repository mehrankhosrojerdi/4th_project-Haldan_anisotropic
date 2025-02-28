import quimb as qu
import quimb.tensor as qtn
import numpy as np
from joblib import Parallel, delayed


class Haldan_anis:

    #-----------------------------------------------------------------------#
    def __init__(self, L, ls):
        self.L = L 
        self.ls = ls 
    #-----------------------------------------------------------------------#
    def MPO(self, D, E):

        J=1
        I = qu.eye(3).real  
        Sx = qu.spin_operator('X', S=1).real
        Sy = qu.spin_operator('Y', S=1).real
        Sz = qu.spin_operator('Z', S=1).real
        
       
        W = np.zeros([6, 6, 3, 3], dtype=float)

        
        W[0, 0, :, :] = I  
        W[0, 1, :, :] = J * Sx
        W[0, 2, :, :] = J * Sy
        W[0, 3, :, :] = J * Sz
        W[0, 5, :, :] = D * (Sz @ Sz) + E * (Sx @ Sx - Sy @ Sy)

        
        W[1, 5, :, :] = Sx
        W[2, 5, :, :] = Sy
        W[3, 5, :, :] = Sz
        W[4, 5, :, :] = I  
        
        
        W[5, 5, :, :] = I

        
        Wl = W[0, :, :, :]  
        Wr = W[:, 5, :, :]  

        H = qtn.MatrixProductOperator([Wl] + [W] * (self.L - 2) + [Wr])

        return H
    #-----------------------------------------------------------------------#
    def P(self):
        I = qu.eye(3).real
        Z = qu.pauli('Z',dim=3).real
        # make projection
        W = np.zeros([2, 2, 3, 3], dtype = float)
        W[0, 0, :, :] = I
        W[1, 1, :, :] = Z
        Wr = np.zeros([2, 3, 3], dtype = float)
        Wr[0, :, :] = I
        Wr[1, :, :] = Z
        Wl = np.zeros([2, 3, 3], dtype = float)
        Wl[0, :, :] = I
        Wl[1, :, :] = -Z
        Wrplus = Wr
        Wlplus = Wr
        Wrminus = Wr
        Wlminus = Wl    

        Identity = np.zeros([2, 2, 3, 3], dtype = int)
        Identity[0, 0, :, :] = I
        Identity[1, 1, :, :] = I
        Identity_side = np.zeros([2, 3, 3], dtype = int)
        Identity_side[0, :, :] = I
        Identity_side[1, :, :] = I
        Identity_side_minus = np.zeros([2, 3, 3], dtype = int)
        Identity_side_minus[0, :, :] = I
        Identity_side_minus[1, :, :] = -I
    

        # build projection odd and even
        
        #if int(self.L) % 2 != 0:

        even_form = [Identity]+[W]
        even_repeat = sum([even_form for i in range(int((self.L-3)/2))],[])

        odd_form = [W]+[Identity]
        odd_repeat = sum([odd_form for i in range(int((self.L-3)/2))],[])

        P_plus_even = qtn.MatrixProductOperator([Wlplus] +  even_repeat + [Identity] + [Wrplus])
        P_plus_odd = qtn.MatrixProductOperator([Identity_side] + odd_repeat + [W] + [Identity_side])
        P_minus_even = qtn.MatrixProductOperator([Wlminus] + even_repeat+ [Identity] + [Wrminus])
        P_minus_odd = qtn.MatrixProductOperator([Identity_side_minus] + odd_repeat + [W] + [Identity_side])

        #elif int(self.L) % 2 == 0:

            #even_form = [Identity]+[W]
            #even_repeat = sum([even_form for i in range(int((self.L-2)/2))],[])

            #odd_form = [W]+[Identity]
            #odd_repeat = sum([odd_form for i in range(int((self.L-2)/2))],[])

            #P_plus_even = qtn.MatrixProductOperator([Wlplus] +  even_repeat + [Identity_side])
            #P_plus_odd = qtn.MatrixProductOperator([Identity_side] + odd_repeat +  [Wrplus])
            #P_minus_even = qtn.MatrixProductOperator([Wlminus] + even_repeat+  [Identity_side] )
            #P_minus_odd = qtn.MatrixProductOperator([Identity_side_minus] + odd_repeat + [Wrminus])

        # build projection
        P_plus = qtn.MatrixProductOperator([Wlplus] + [W] * (self.L - 2) + [Wrplus])
        P_minus = qtn.MatrixProductOperator([Wlminus] + [W] * (self.L - 2) + [Wrminus])

        return P_plus_even, P_plus_odd, P_minus_even, P_minus_odd, P_plus, P_minus
    #-----------------------------------------------------------------------#
    def DMRG(self, d1, e1):
        DMRG = qtn.tensor_dmrg.DMRG(ham = Haldan_anis(L = self.L, ls = self.ls).MPO(D = d1, E = e1), bond_dims = 150) 
        DMRG.solve(tol = 1e-3, verbosity = 0);
        ground_state = DMRG.state
        energy = DMRG.energy
        return ground_state, energy
    #-----------------------------------------------------------------------#
    def apply_projection(self, DMRG_state, target_value, projections):
        return [(proj.apply(DMRG_state), target_value) for proj in projections]

    def process_point(self, d, e, target_value, projections):
        DMRG_state, _ = Haldan_anis(L=self.L, ls=self.ls).DMRG(d1=d, e1=e)
        projections_result = self.apply_projection(DMRG_state, target_value, projections)
        return ([-2, e], DMRG_state, target_value, projections_result)

    def generate_train_set(self, n_jobs=5):
        E = np.arange(-2, 2, 0.1)
        D = np.arange(-2, 2, 0.1)
        
        projections = Haldan_anis(L=self.L, ls=self.ls).P()[:4]  # Extract relevant projection operators
        
        conditions = [
            (lambda e: 0.8 <= e <= 2, -2, 1),
            (lambda e: -0.8 <= e < 0.8, -2, 3),
            (lambda e: -2 <= e < -0.8, -2, 2),
            (lambda e: -2 < e < -0.4, 2, 4),
            (lambda e: -0.4 < e < 0.4, 2, 5),
            (lambda e: 0.4 < e < 2.0, 2, 6),
        ]
        
        d_conditions = [
            (lambda d: -2 < d < 0.2, 2, 1),
            (lambda d: 0.2 < d < 2.0, 2, 6),
            (lambda d: -2 < d < 0.2, -2, 2),
            (lambda d: 0.2 < d < 2.0, -2, 4),
            (lambda d: -2 < d < -0.5, 0.0, 3),
            (lambda d: 0.9 < d < 2, 0.0, 5),
            (lambda d: -0.2 < d < 0.6, 0.0, 7),
        ]
        
        # Parallel execution over E
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.process_point)(d1, e, target, projections)
            for condition, d1, target in conditions
            for e in E if condition(e)
        )
        
        # Parallel execution over D
        results += Parallel(n_jobs=n_jobs)(
            delayed(self.process_point)(d, e1, target, projections)
            for condition, d, target in d_conditions
            for e1 in [2, -2, 0.0] if condition(d)
        )
        
        # Unpack results
        lst_points, lst_DMRG, lst_target, lst_contract = [], [], [], []
        
        for points, state, target, projections in results:
            lst_points.append(points)
            lst_DMRG.append(state)
            lst_target.append(target)
            for proj_state, proj_target in projections:
                lst_contract.append((proj_state, proj_target))
        
        return (
            np.array(lst_DMRG),
            np.array(lst_target),
            np.array([p[0] for p in lst_contract]),
            np.array([p[1] for p in lst_contract]),
            np.array(lst_points),
        )

    
    def generate_test_set(self):
        E = np.arange(-2, 2, 0.1)
        D = np.arange(-2, 2, 0.1)
        
        def compute_dmrg(d, e):
            return [d, e], Haldan_anis(L = self.L, ls = self.ls).DMRG(d1=d, e1=e)[0]

        results = Parallel(n_jobs=5, backend = 'loky')(delayed(compute_dmrg)(d, e) for e in E for d in D)
        
        lst_points, lst_DMRG = zip(*results)
        
        return np.array(lst_DMRG), np.array(lst_points)