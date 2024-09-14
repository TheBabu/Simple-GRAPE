from qiskit_nature.second_q.operators import SpinOp

class SimpleGRAPE:
    def __init__(self, hilbert_dimension, num_of_intervals, total_time, drift_parameter, initial_state, target_state):
        self.spin             = (hilbert_dimension - 1) / 2
        self.num_of_intervals = num_of_intervals
        self.total_time       = total_time
        self.drift_parameter  = drift_parameter
        self.time_step        = total_time / num_of_intervals
        self.initial_state    = initial_state
        self.target_state     = target_state

        self.x_spin_operator         = SpinOp.x(self.spin)
        self.y_spin_operator         = SpinOp.y(self.spin)
        self.z_spin_operator         = SpinOp.z(self.spin)
        self.z_squared_spin_operator = SpinOp({"Z_0 Z_0": 1}, self.spin)
        self.squared_spin_operator   = SpinOp({"X_0 X_0": 1, "Y_0 Y_0": 1, "Z_0 Z_0": 1}, self.spin) #Necessary?

    def run(self):
        fidelity_list = []

        return fidelity_list

