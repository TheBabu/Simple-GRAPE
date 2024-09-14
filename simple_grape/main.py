#!/usr/bin/env python3

from qiskit_nature.second_q.operators import SpinOp
from qiskit.quantum_info import Statevector, random_statevector, state_fidelity
from qiskit.circuit import Parameter
import numpy as np
import scipy as sp

from simple_grape import SimpleGRAPE

if __name__ == "__main__":
    HILBERT_DIMENSION = 2
    NUM_OF_INTERVALS  = 100
    TOTAL_TIME        = 3
    DRIFT_PARAMETER   = 3
    
    spin                    = (HILBERT_DIMENSION - 1) / 2
    x_spin_operator         = SpinOp.x(spin)
    y_spin_operator         = SpinOp.y(spin)
    z_spin_operator         = SpinOp.z(spin)
    z_squared_spin_operator = SpinOp({"Z_0 Z_0": 1}, spin)
    squared_spin_operator   = SpinOp({"X_0 X_0": 1, "Y_0 Y_0": 1, "Z_0 Z_0": 1}, spin) #Necessary?
    
    eigenvalues, eigenvectors = sp.linalg.eig(z_spin_operator.to_matrix())
    min_stretched_eigenvalue  = eigenvalues[-1]
    min_stretched_state       = eigenvectors[-1]
    max_stretched_eigenvalue  = eigenvalues[0]
    max_stretched_state       = eigenvectors[0]
    
    initial_state = Statevector(max_stretched_state) #Could just use Statevector.from_int?
    target_state  = random_statevector(HILBERT_DIMENSION, seed=0)
    time_step     = TOTAL_TIME / NUM_OF_INTERVALS
    
    #TODO: Fix qiskit_nature casting
    x_term_parameter = Parameter("theta_x")
    y_term_parameter = Parameter("theta_y")
    
    hamiltonian_x_term     = np.cos(x_term_parameter) * x_spin_operator
    hamiltonian_y_term     = np.sin(y_term_parameter) * y_spin_operator
    hamiltonian_drift_term = DRIFT_PARAMETER * z_squared_spin_operator
    
    control_waveform_x_terms = [1] * NUM_OF_INTERVALS
    control_waveform_y_terms = [1] * NUM_OF_INTERVALS

#TODO: Cleanup with OOP
def update_unitary_terms():
    unitary_terms = []

    for i in range(NUM_OF_INTERVALS):
        hamiltonian_x_term_assigned = hamiltonian_x_term.assign_parameters({ x_term_parameter: control_waveform_x_terms[i]})
        hamiltonian_y_term_assigned = hamiltonian_y_term.assign_parameters({ y_term_parameter: control_waveform_y_terms[i]})

        hamiltonian_assigned = (hamiltonian_drift_term + hamiltonian_x_term_assigned + hamiltonian_y_term_assigned).to_matrix()
        # print(hamiltonian_assigned)
        unitary_term = sp.linalg.expm(hamiltonian_assigned)

        # unitary_terms.append(unitary_term)

    return unitary_terms

def calculate_fidelity(unitary, target_state, initial_state):
    evolved_state = initial_state.evolve(unitary)
    
    fidelity = state_fidelity(target_state, evolved_state)

    return fidelity

def calculate_gradient():
    pass

unitary_terms = update_unitary_terms()

