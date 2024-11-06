from qiskit_nature.second_q.operators import SpinOp
from qiskit.quantum_info import state_fidelity, Operator
import numpy as np
import scipy as sp

class SimpleGRAPE:
    def __init__(self, hilbert_dimension, num_of_intervals, total_time, drift_parameter, truncated_taylor_len, init_seed, initial_state, target_state, check_grad):
        self.hilbert_dimension    = hilbert_dimension
        self.spin                 = (hilbert_dimension - 1) / 2
        self.num_of_intervals     = num_of_intervals #N
        self.total_time           = total_time #T
        self.drift_parameter      = drift_parameter #beta
        self.truncated_taylor_len = truncated_taylor_len
        self.init_seed            = init_seed
        self.time_step            = total_time / num_of_intervals #delta t
        self.initial_state        = initial_state #psi 0
        self.target_state         = target_state #psi targ
        self.check_grad           = check_grad

        #Initialize spin operators
        self.x_spin_operator        = Operator(SpinOp.x(self.spin)) #Jx
        self.y_spin_operator        = Operator(SpinOp.y(self.spin)) #Jy
        z_squared_spin_operator     = Operator(SpinOp({"Z_0 Z_0": 1}, self.spin)) #Jz^2
        self.drift_hamiltonian_term = self.drift_parameter * z_squared_spin_operator #beta * Jz^2

        #Optimization data
        self.states    = []
        self.cost_list = []

    def compose_unitaries(self, unitary_list):
        total_unitary = Operator(np.eye(self.hilbert_dimension))

        for unitary in unitary_list:
            total_unitary = total_unitary.compose(unitary)

        return total_unitary

    def generate_discrete_hamiltonian(self, theta):
        x_spin_hamiltonian_term = np.cos(theta) * self.x_spin_operator
        y_spin_hamiltonian_term = np.sin(theta) * self.y_spin_operator
        discrete_hamiltonian    = self.drift_hamiltonian_term + x_spin_hamiltonian_term + y_spin_hamiltonian_term

        return discrete_hamiltonian

    def generate_unitary_list(self, theta_waveforms):
        unitary_list = [
            Operator(sp.linalg.expm(-1j * self.time_step * self.generate_discrete_hamiltonian(theta)))
            for theta in theta_waveforms
        ]

        return unitary_list

    def calculate_cost_and_gradient(self, theta_waveforms, trunacted_taylor_length):
        #Generate list of all unitaries at each time with specific ukj
        unitary_list = self.generate_unitary_list(theta_waveforms)

        #Create total unitary
        total_unitary = self.compose_unitaries(unitary_list) #U_tot = U_N U_N-1 U_N-2 ... U_j ... U_2 U_1

        #Calculate cost
        evolved_state = self.initial_state.evolve(total_unitary) #U_tot|psi_0>
        cost          = -1 * state_fidelity(evolved_state, self.target_state) #-F

        #Save optimization data
        self.states.append(evolved_state)
        self.cost_list.append(cost)

        #Calculate gradient
        cost_gradient = [] #del C / del uj

        #Calculate the first part of the gradient
        complex_conjugate_expectation = self.target_state.inner(evolved_state).conj() #<psi_t|U_tot|psi_0>*

        #Calculate gradient for x_spin and y_spin term
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

            evolved_state = self.initial_state.evolve(composed_unitary) #U|psi_0>

            composed_expectation = self.target_state.inner(evolved_state) #<psi_t|U|psi_0>

            #Update cost gradient (for specific time step)
            cost_gradient.append(-2 * np.real(complex_conjugate_expectation * composed_expectation)) #-2 Re{<psi_t|U_tot|psi_0>* <psi_t|U|psi_0>}

        return cost, cost_gradient
    
    def run(self):
        cost_and_gradient_func = lambda waveforms : (
            self.calculate_cost_and_gradient(waveforms[0:self.num_of_intervals], self.truncated_taylor_len)
        )

        #Initialize theta waveforms randomly from -pi to pi
        rng                     = np.random.default_rng(self.init_seed) #Set seed for reproducibility
        initial_theta_waveforms = [rng.uniform(low=-np.pi, high=np.pi) for _ in range(self.num_of_intervals)]

        #If check grad, return gradient error compared to finite differences at initial waveforms (Don't perform classical optimization)
        if(self.check_grad):
            #A bit hacky
            cost_func = lambda waveforms : cost_and_gradient_func(waveforms)[0]
            grad_func = lambda waveforms : cost_and_gradient_func(waveforms)[1]

            grad_error = sp.optimize.check_grad(cost_func, grad_func, initial_theta_waveforms)

            return grad_error

        #Perform classical optimization
        bounds           = [(-np.pi, np.pi)] * (self.num_of_intervals)
        optimizer_result = sp.optimize.minimize(cost_and_gradient_func,
                                                x0=initial_theta_waveforms,
                                                jac=True,
                                                method="L-BFGS-B",
                                                bounds=bounds)

        final_cost        = optimizer_result.fun
        theta_waveforms = optimizer_result.x[0:self.num_of_intervals]
        unitary_list      = self.generate_unitary_list(theta_waveforms)

        return (final_cost, theta_waveforms, unitary_list)

