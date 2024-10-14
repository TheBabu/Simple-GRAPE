from qiskit_nature.second_q.operators import SpinOp
from qiskit.quantum_info import state_fidelity, Operator
import numpy as np
import scipy as sp

class SimpleGRAPE:
    def __init__(self, hilbert_dimension, num_of_intervals, total_time, drift_parameter, truncated_taylor_len, init_seed, initial_state, target_state):
        self.hilbert_dimension    = hilbert_dimension
        self.spin                 = (hilbert_dimension - 1) / 2
        self.num_of_intervals     = num_of_intervals #N
        self.total_time           = total_time #T
        self.drift_parameter      = drift_parameter #beta
        self.truncated_taylor_len = truncated_taylor_len
        self.init_seed            = init_seed
        self.time_step            = total_time / (num_of_intervals - 1) #delta t
        self.initial_state        = initial_state #psi 0
        self.target_state         = target_state #psi targ

        #Initialize spin operators
        self.x_spin_operator        = Operator(SpinOp.x(self.spin)) #Jx
        self.y_spin_operator        = Operator(SpinOp.y(self.spin)) #Jy
        z_squared_spin_operator     = Operator(SpinOp({"Z_0 Z_0": 1}, self.spin)) #Jz^2
        self.drift_hamiltonian_term = self.drift_parameter * z_squared_spin_operator

    def compose_unitaries(self, unitary_list):
        total_unitary = Operator(np.eye(self.hilbert_dimension))

        for unitary in unitary_list:
            total_unitary = total_unitary.compose(unitary)

        return total_unitary

    def generate_discrete_hamiltonian(self, theta_x, theta_y):
        x_spin_hamiltonian_term = np.cos(theta_x) * self.x_spin_operator
        y_spin_hamiltonian_term = np.sin(theta_y) * self.y_spin_operator
        discrete_hamiltonian    = self.drift_hamiltonian_term + x_spin_hamiltonian_term + y_spin_hamiltonian_term

        return discrete_hamiltonian

    def generate_unitary_list(self, theta_x_waveforms, theta_y_waveforms):
        unitary_list = [
            sp.linalg.expm(-1j * self.time_step * self.generate_discrete_hamiltonian(theta_x_waveforms[i], theta_y_waveforms[i]))
            for i in range(self.num_of_intervals)
        ]

        return unitary_list

    def calculate_cost_and_gradient(self, theta_x_waveforms, theta_y_waveforms, trunacted_taylor_length):
        #Generate list of all unitaries at each time with specific ukj
        unitary_list = self.generate_unitary_list(theta_x_waveforms, theta_y_waveforms)

        #Create total unitary
        total_unitary = self.compose_unitaries(unitary_list)

        #Calculate cost
        evolved_state = self.initial_state.evolve(total_unitary)
        cost          = -1 * state_fidelity(evolved_state, self.target_state)

        #Calculate gradient
        x_spin_gradient = []
        y_spin_gradient = []

        #Calculate the first part of the gradient
        evolved_state                 = self.initial_state.evolve(total_unitary)
        complex_conjugate_expectation = self.target_state.inner(evolved_state).conj()

        #Calculate gradient for x_spin and y_spin term
        for i, (theta_x, theta_y) in enumerate(zip(theta_x_waveforms, theta_y_waveforms)):
            x_spin_partial_derivative = Operator(np.eye(self.hilbert_dimension))
            y_spin_partial_derivative = Operator(np.eye(self.hilbert_dimension))

            for n in range(trunacted_taylor_length):
                discrete_hamiltonian = self.generate_discrete_hamiltonian(theta_x, theta_y)

                x_spin_partial_derivative +=\
                    ((-1j) ** n) / (sp.special.factorial(n)) *\
                    sum((discrete_hamiltonian ** (i - 1)) @ self.x_spin_operator @ (discrete_hamiltonian ** (n - 1)) for i in range(1, n + 1))
                y_spin_partial_derivative +=\
                    ((-1j) ** n) / (sp.special.factorial(n)) *\
                    sum((discrete_hamiltonian ** (i - 1)) @ self.y_spin_operator @ (discrete_hamiltonian ** (n - 1)) for i in range(1, n + 1))

            front_slice_unitaries = unitary_list[0:i]
            back_slice_unitaries  = unitary_list[i + 1: self.num_of_intervals]

            front_slice_total_unitary = self.compose_unitaries(front_slice_unitaries)
            back_slice_total_unitary  = self.compose_unitaries(back_slice_unitaries)

            x_spin_composed_unitary = front_slice_total_unitary.compose(x_spin_partial_derivative.compose(back_slice_total_unitary))
            y_spin_composed_unitary = front_slice_total_unitary.compose(y_spin_partial_derivative.compose(back_slice_total_unitary))

            x_spin_evolved_state = self.initial_state.evolve(x_spin_composed_unitary)

            y_spin_evolved_state = self.initial_state.evolve(y_spin_composed_unitary)

            x_spin_composed_expectation = self.target_state.inner(x_spin_evolved_state)
            y_spin_composed_expectation = self.target_state.inner(y_spin_evolved_state)

            x_spin_gradient.append(2 * np.real(complex_conjugate_expectation * x_spin_composed_expectation))
            y_spin_gradient.append(2 * np.real(complex_conjugate_expectation * y_spin_composed_expectation))

        #Compose x_spin and y_spin gradients together
        gradient = x_spin_gradient + y_spin_gradient

        return cost, gradient
    
    def run(self):
        #Slice waveforms for x_spin and y_spin operators
        cost_and_gradient_func = lambda waveforms : (
            self.calculate_cost_and_gradient(waveforms[0:self.num_of_intervals],
                                             waveforms[self.num_of_intervals:self.num_of_intervals * 2],
                                             self.truncated_taylor_len)
        )

        #Initialize theta waveforms randomly from -pi to pi
        rng                     = np.random.default_rng(self.init_seed) #Set seed for reproducibility
        initial_theta_waveforms = [rng.uniform(low=-np.pi, high=np.pi) for _ in range(2 * self.num_of_intervals)]

        #Perform classical optimization
        bounds           = [(-np.pi, np.pi)] * (2 * self.num_of_intervals)
        optimizer_result = sp.optimize.minimize(cost_and_gradient_func,
                                                x0=initial_theta_waveforms,
                                                jac=True,
                                                method="L-BFGS-B",
                                                bounds=bounds)

        #DEBUG
        print(optimizer_result)

        return (optimizer_result.fun, optimizer_result.x[0:self.num_of_intervals], optimizer_result.x[self.num_of_intervals:self.num_of_intervals * 2])

