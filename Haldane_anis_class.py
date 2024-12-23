import quimb as qu
import quimb.tensor as qtn
import numpy as np

class Haldan_anis:

    #-----------------------------------------------------------------------#
    def __init__(self, L, ls):
        self.L = L # number of particle
        self.ls = ls # the scale of dividing the range of h and k
    #-----------------------------------------------------------------------#
    def MPO(self, d, e):
        I = qu.eye(3)
        Z = qu.pauli('Z',dim=3)
        X = qu.pauli('X',dim=3)
        Y = qu.pauli('y',dim=3) 
        S = np.array([X,Y,Z])
        res= (S@S)
    
        # define the MPO tensor
        W = np.zeros([5, 5, 3, 3], dtype = float)
        # allocate different values to each of the sites
        W[0, 0, :, :] = I
        W[0, 1, :, :] = res[0]+res[1]+res[2]
        W[0, 4, :, :] = (d*Z@Z)+(e*X@X)-(e*Y@Y)
        W[1, 4, :, :] = I
        W[4, 4, :, :] = I
        Wl = W[0, :, :, :]
        Wr = W[:, 4, :, :]
        # build the MPO
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

        # make identity
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
        DMRG = qtn.tensor_dmrg.DMRG(ham = Haldan_anis(L = self.L, ls = self.ls).MPO(d = d1, e = e1), bond_dims = 150) 
        DMRG.solve(tol = 1e-3, verbosity = 0);
        ground_state = DMRG.state
        energy = DMRG.energy
        return ground_state, energy
    #-----------------------------------------------------------------------#
    def generate_train_set(self):

        # Define ranges for E and D
        E = np.arange(-2, 2, 0.1)
        D = np.arange(-2, 2, 0.1)

        # Lists to store points and targets
        lst_points = []
        lst_DMRG = []
        lst_target = []
        lst_contract = []
        lst_target_projection = []

        # make projections
        P_plus_even, P_plus_odd, P_minus_even, P_minus_odd, _, _ = Haldan_anis(L = self.L, ls = self.ls).P()

        def apply_projection(DMRG_state, target_value):
            results = []
            for projection in[P_plus_even, P_plus_odd, P_minus_even, P_minus_odd]:
                contraction_state = projection.apply(DMRG_state)
                results.append((contraction_state, target_value))
            return results

        # Loop through E and D and classify points
        for e in E:
            if 0.8 <= e <= 2:
                target_value = 1
                lst_points.append([-2, e])
                DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = -2, e1 = e)
                lst_DMRG.append(DMRG_state) # DMRG states
                lst_target.append(target_value)  # 'large_ex'
                projection = apply_projection(DMRG_state, target_value) # making the projection states
                for state, t_value in projection:
                    lst_contract.append(state)
                    lst_target_projection.append(t_value)
               
            elif -0.8 <= e < 0.8:
                lst_points.append([-2, e])
                target_value = 3
                DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = -2, e1 = e)
                lst_DMRG.append(DMRG_state) # DMRG states
                lst_target.append(target_value)  # 'z_neel'
                projection = apply_projection(DMRG_state, target_value) # making the projection states
                for state, t_value in projection:
                    lst_contract.append(state)
                    lst_target_projection.append(t_value)

            elif -2 <= e < -0.8:
                lst_points.append([-2, e])
                target_value = 2
                DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = -2, e1 = e)
                lst_DMRG.append(DMRG_state) # DMRG states
                lst_target.append(target_value)  # 'large_ey'
                projection = apply_projection(DMRG_state, target_value) # making the projection states
                for state, t_value in projection:
                    lst_contract.append(state)
                    lst_target_projection.append(t_value)

        for e in E:
            if -2 < e < -0.4:
                lst_points.append([2, e])
                target_value = 4
                DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = 2, e1 = e)
                lst_DMRG.append(DMRG_state) # DMRG states
                lst_target.append(target_value)  # 'x_neel'
                projection = apply_projection(DMRG_state, target_value) # making the projection states
                for state, t_value in projection:
                    lst_contract.append(state)
                    lst_target_projection.append(t_value)

            elif -0.4 < e < 0.4:
                lst_points.append([2, e])
                target_value = 5
                DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = 2, e1 = e)
                lst_DMRG.append(DMRG_state) # DMRG states
                lst_target.append(target_value)  # 'large_d'
                projection = apply_projection(DMRG_state, target_value) # making the projection states
                for state, t_value in projection:
                    lst_contract.append(state)
                    lst_target_projection.append(t_value)

            elif 0.4 < e < 2.0:
                lst_points.append([2, e])
                target_value = 6
                DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = 2, e1 = e)
                lst_DMRG.append(DMRG_state) # DMRG states
                lst_target.append(target_value)  # 'y_neel'
                projection = apply_projection(DMRG_state, target_value) # making the projection states
                for state, t_value in projection:
                    lst_contract.append(state)
                    lst_target_projection.append(t_value)

        for d in D:
            if -2 < d < 0.2:
                lst_points.append([d, 2])
                target_value = 1
                DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = d, e1 = 2)
                lst_DMRG.append(DMRG_state) # DMRG states
                lst_target.append(target_value)  # 'large_ex'
                projection = apply_projection(DMRG_state, target_value) # making the projection states
                for state, t_value in projection:
                    lst_contract.append(state)
                    lst_target_projection.append(t_value)

            elif 0.2 < d < 2.0:
                lst_points.append([d, 2])
                target_value = 6
                DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = d, e1 = 2)
                lst_DMRG.append(DMRG_state) # DMRG states
                lst_target.append(target_value)  # 'y_neel'
                projection = apply_projection(DMRG_state, target_value) # making the projection states
                for state, t_value in projection:
                    lst_contract.append(state)
                    lst_target_projection.append(t_value)

        for d in D:
            if -2 < d < 0.2:
                lst_points.append([d, -2])
                target_value = 2
                DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = d, e1 = -2)
                lst_DMRG.append(DMRG_state) # DMRG states
                lst_target.append(target_value)  # 'large_ey'
                projection = apply_projection(DMRG_state, target_value) # making the projection states
                for state, t_value in projection:
                    lst_contract.append(state)
                    lst_target_projection.append(t_value)

            elif 0.2 < d < 2.0:
                lst_points.append([d, -2])
                target_value = 4
                DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = d, e1 = -2)
                lst_DMRG.append(DMRG_state) # DMRG states
                lst_target.append(target_value)  # 'x_neel'
                projection = apply_projection(DMRG_state, target_value) # making the projection states
                for state, t_value in projection:
                    lst_contract.append(state)
                    lst_target_projection.append(t_value)
            
        for d in np.arange(-2, -0.5, 0.1):  # Added step size 0.1
            lst_points.append([d, 0.0])
            target_value = 3
            DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = d, e1 = 0.0)
            lst_DMRG.append(DMRG_state) # DMRG states
            lst_target.append(target_value)
            projection = apply_projection(DMRG_state, target_value) # making the projection states
            for state, t_value in projection:
                lst_contract.append(state)
                lst_target_projection.append(t_value)

        for d in np.arange(0.9, 2, 0.1):
            lst_points.append([d, 0.0])
            target_value = 5
            DMRG_state, DMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = d, e1 = 0.0)
            lst_DMRG.append(DMRG_state) # DMRG states
            lst_target.append(target_value)  # 'large_d'
            projection = apply_projection(DMRG_state, target_value) # making the projection states
            for state, t_value in projection:
                lst_contract.append(state)
                lst_target_projection.append(t_value)

        for d in np.arange(-0.2, 0.6, 0.1):
            lst_points.append([d, 0.0])
            target_value = 7
            DMRG_state, lst_contractDMRG_energy = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = d, e1 = 0.0)
            lst_DMRG.append(DMRG_state) # DMRG states
            lst_target.append(target_value) #'Haldane'
            projection = apply_projection(DMRG_state, target_value) # making the projection states
            for state, t_value in projection:
                lst_contract.append(state)
                lst_target_projection.append(t_value)

        DMRG_state = np.array(lst_DMRG)
        DMRG_target = np.array(lst_target)
        points = np.array(lst_points)
        project_state = np.array(lst_contract)
        projection_target = np.array(lst_target_projection)

        return DMRG_state, DMRG_target, project_state, projection_target, points
    

    def generate_test_set(self):
        # Generate the dataset for the specified constant h1
        E = np.arange(-2, 2, 0.1)
        D = np.arange(-2, 2, 0.1)
        lst_points = []
        lst_DMRG = []
        for e in E:
            for d in D:
                lst_points.append([d,e])
                DMRG_state, _ = Haldan_anis(L = self.L, ls = self.ls).DMRG(d1 = d, e1 = e); # make DMRG state for these specific value of h and k
                lst_DMRG.append(DMRG_state) # DMRG states


        DMRG_state = np.array(lst_DMRG)
        points = np.array(lst_points)

        return DMRG_state, points