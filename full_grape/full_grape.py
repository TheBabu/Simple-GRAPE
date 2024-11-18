from qiskit_nature.second_q.operators import SpinOp
from qiskit.quantum_info import Operator
import numpy as np
import scipy as sp

from simple_grape.simple_grape import SimpleGRAPE

class FullGRAPE(SimpleGRAPE):
    def __init__(self, hilbert_dimension, num_of_targets, num_of_intervals, total_time, drift_parameter, truncated_taylor_len, init_seed, initial_states, target_states, check_grad):
        self.hilbert_dimension    = hilbert_dimension
        self.spin                 = (hilbert_dimension - 1) / 2
        self.num_of_targets       = num_of_targets #K
        self.num_of_intervals     = num_of_intervals #N
        self.total_time           = total_time #T
        self.drift_parameter      = drift_parameter #beta
        self.truncated_taylor_len = truncated_taylor_len
        self.init_seed            = init_seed
        self.time_step            = total_time / num_of_intervals #delta t
        self.initial_states       = initial_states #{psi_i}
        self.target_states        = target_states #{psi_t,k}
        self.check_grad           = check_grad

        if(len(initial_states) > hilbert_dimension):
            raise Exception("Too many initial states")
        if(len(target_states) > hilbert_dimension):
            raise Exception("Too many target states")
        if(len(target_states) != len(initial_states)):
            raise Exception("Number of initial states do not match to number of target states")

        #Create target operator
        self.target_operator = Operator(
            sum(
                np.outer(target_state, initial_state.conjugate()) 
                for target_state, initial_state in zip(target_states, initial_states)
            )
        )

        #Initialize spin operators
        self.x_spin_operator        = Operator(SpinOp.x(self.spin)) #Jx
        self.y_spin_operator        = Operator(SpinOp.y(self.spin)) #Jy
        z_squared_spin_operator     = Operator(SpinOp({"Z_0 Z_0": 1}, self.spin)) #Jz^2
        self.drift_hamiltonian_term = self.drift_parameter * z_squared_spin_operator #beta * Jz^2

    def calculate_cost_and_gradient(self, theta_waveforms, trunacted_taylor_length):
        #Generate list of all unitaries at each time with specific ukj
        unitary_list = self.generate_unitary_list(theta_waveforms)

        #Create total unitary
        total_unitary = self.compose_unitaries(unitary_list) #U_tot = U_N U_N-1 U_N-2 ... U_j ... U_2 U_1

        #Calculate cost
        cost = -1 * (np.abs(np.trace(total_unitary.compose(self.target_operator.adjoint()))) ** 2) / (self.num_of_targets ** 2) #-F

        #Calculate gradient
        cost_gradient = [] #del C / del uj

        #Calculate the first part of the gradient
        evolved_total_states = [initial_state.evolve(total_unitary) for initial_state in self.initial_states] #{U_tot|psi_k>}
        complex_conjugate_expectations = [
            target_state.inner(evolved_state).conj()
            for target_state, evolved_state in zip(self.target_states, evolved_total_states)
        ] #{<psi_t|U_tot|psi_i>*}

        #Calculate gradient
        for j, theta in enumerate(theta_waveforms):
            #Calculate partial derivatives
            unitary_gradient = Operator(np.zeros((self.hilbert_dimension, self.hilbert_dimension))) #del Uj / del uj

            for n in range(trunacted_taylor_length):
                discrete_hamiltonian = self.generate_discrete_hamiltonian(theta) #Hj

                unitary_gradient +=\
                    ((-1j * self.time_step) ** n) / (sp.special.factorial(n)) *\
                    sum(
                        (discrete_hamiltonian ** (i - 1)) @
                        ((-np.sin(theta) * self.x_spin_operator) + (np.cos(theta) * self.y_spin_operator)) @
                        (discrete_hamiltonian ** (n - i))
                    for i in range(1, n + 1))

            #Compose front and back slice unitaries
            front_slice_unitaries = unitary_list[0:j]
            back_slice_unitaries  = unitary_list[j + 1: self.num_of_intervals]

            front_slice_total_unitary = self.compose_unitaries(front_slice_unitaries) #U_j-1 ... U_2 U_1
            back_slice_total_unitary  = self.compose_unitaries(back_slice_unitaries) #U_N U_N-1 ... U_j+1

            #U = U_N U_N-1 U_N-2 ... del U_j / del u1j ... U_2 U_1
            composed_unitary = front_slice_total_unitary.compose(unitary_gradient).compose(back_slice_total_unitary)

            evolved_states = [initial_state.evolve(composed_unitary) for initial_state in self.initial_states]  #{U|psi_k>}

            composed_expectations = [
                target_state.inner(evolved_state)
                for target_state, evolved_state in zip(self.target_states, evolved_states)
            ] #{<psi_t|U|psi_k>}

            #Update cost gradient (for specific time step)
            cost_gradient_jth_time_step = 0
            for complex_conjugate_expectation in complex_conjugate_expectations:
                for composed_expectation in composed_expectations:
                    cost_gradient_jth_time_step += np.real(complex_conjugate_expectation * composed_expectation)
            cost_gradient_jth_time_step *= -2 / (self.num_of_targets ** 2) #(-2 / K) * Sum_m,n Re{<psi_t,m|U_tot|psi_m>* <psi_t,n|U|psi_n>}

            cost_gradient.append(cost_gradient_jth_time_step) 

        return cost, cost_gradient

