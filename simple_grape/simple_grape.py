from qiskit_nature.second_q.operators import SpinOp
from qiskit.circuit import Parameter
from qiskit.quantum_info import state_fidelity, Operator
import numpy as np
import scipy as sp

class SimpleGRAPE:
    def __init__(self, hilbert_dimension, num_of_intervals, total_time, drift_parameter, convergence_limit, initial_state, target_state):
        self.hilbert_dimension = hilbert_dimension
        self.spin              = (hilbert_dimension - 1) / 2
        self.num_of_intervals  = num_of_intervals #N
        self.total_time        = total_time #T
        self.drift_parameter   = drift_parameter #beta
        self.convergence_limit = convergence_limit
        self.time_step         = total_time / num_of_intervals #delta t
        self.initial_state     = initial_state #psi 0
        self.target_state      = target_state #psi targ

        #Initialize spin operators
        self.x_spin_operator         = SpinOp.x(self.spin) #Jx
        self.y_spin_operator         = SpinOp.y(self.spin) #Jy
        z_squared_spin_operator = SpinOp({"Z_0 Z_0": 1}, self.spin) #Jz^2

        #Initialize Hamiltonian terms (K = 2)
        self.x_parameter = Parameter("theta_x")
        self.y_parameter = Parameter("theta_y")

        self.drift_hamiltonian_term  = self.drift_parameter * z_squared_spin_operator
        self.x_spin_hamiltonian_term = np.cos(self.x_parameter) * self.x_spin_operator
        self.y_spin_hamiltonian_term = np.sin(self.y_parameter) * self.y_spin_operator

    def compose_unitaries(self, unitary_list):
        total_unitary = Operator(np.eye(self.hilbert_dimension))

        for unitary in unitary_list:
            total_unitary = total_unitary.compose(unitary)

        return total_unitary

    def generate_discrete_hamiltonians(self, theta_x, theta_y):
        discrete_hamiltonian = \
            self.drift_hamiltonian_term + \
            self.x_spin_hamiltonian_term.assign_parameters({ self.x_parameter: theta_x }) + \
            self.y_spin_hamiltonian_term.assign_parameters({ self.y_parameter: theta_y })

        return discrete_hamiltonian

    def generate_unitary_list(self, theta_x_waveforms, theta_y_waveforms):
        unitary_list = [
            sp.linalg.expm(-1j * self.time_step * Operator(self.generate_discrete_hamiltonians(theta_x_waveforms[i], theta_y_waveforms[i])))
            for i in range(self.num_of_intervals)
        ]

        return unitary_list

    def calculate_cost(self, theta_x_waveforms, theta_y_waveforms):
        #Generate list of all unitaries at each time with specific ukj
        unitary_list = self.generate_unitary_list(theta_x_waveforms, theta_y_waveforms)

        #Create total unitary
        total_unitary = self.compose_unitaries(unitary_list)

        #Create evolved state
        evolved_state = self.initial_state.evolve(total_unitary)

        #Calculate fidelity
        fidelity = state_fidelity(evolved_state, self.target_state)

        return -1 * fidelity

    def calculate_gradient(self, theta_x_waveforms, theta_y_waveforms):
        #Generate list of all unitaries at each time with specific ukj
        unitary_list = self.generate_unitary_list(theta_x_waveforms, theta_y_waveforms)

        #Create total unitary
        total_unitary = self.compose_unitaries(unitary_list)

        #Calculate partial deriatives for x spin term
        x_spin_partial_derivatives = [
            (-1j * self.time_step * total_unitary).compose(-1 * np.sin(theta_x_waveform) * self.x_spin_operator, front=True)
            for theta_x_waveform in theta_x_waveforms
        ]

        #Calculate partial derivatives for y spin term
        y_spin_partial_derivatives = [
            (-1j * self.time_step * total_unitary).compose(np.cos(theta_y_waveform) * self.y_spin_operator, front=True)
            for theta_y_waveform in theta_y_waveforms
        ]

        #Calculate the first part of the gradient
        evolved_state                 = self.initial_state.evolve(total_unitary)
        complex_conjugate_expectation = self.target_state.inner(evolved_state).conj()

        #Calculate gradient for x spin term and y spin term
        x_spin_gradient = []
        y_spin_gradient = []
        for i, unitary in enumerate(unitary_list):
            x_spin_partial_derivative = x_spin_partial_derivatives[i]
            y_spin_partial_derivative = y_spin_partial_derivatives[i]

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

        return x_spin_gradient + y_spin_gradient
    
    def run(self):
        #Perform classical optimization
        cost_and_gradient_func = lambda waveforms : (
            self.calculate_cost(waveforms[0:self.num_of_intervals], waveforms[self.num_of_intervals:self.num_of_intervals * 2]),
            self.calculate_gradient(waveforms[0:self.num_of_intervals], waveforms[self.num_of_intervals:self.num_of_intervals * 2])
        ) #Return tuple of cost and gradient values

        initial_theta_x_waveforms = [2] * self.num_of_intervals #Initialize theta_x_waveforms to 0 (TODO: Random initialization?)
        initial_theta_y_waveforms = [2] * self.num_of_intervals #Initialize theta_y_waveforms to 0 (TODO: Random initialization?)
        optimizer_result = sp.optimize.minimize(cost_and_gradient_func, x0=(initial_theta_x_waveforms + initial_theta_y_waveforms), jac=True, method="BFGS")

        #DEBUG
        print(optimizer_result)

        return (optimizer_result.x[0:self.num_of_intervals], optimizer_result.x[self.num_of_intervals:self.num_of_intervals * 2])

