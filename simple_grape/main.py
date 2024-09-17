#!/usr/bin/env python3

from qiskit.quantum_info import Statevector, random_statevector
import argparse

from simple_grape import SimpleGRAPE

if __name__ == "__main__":
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hilbert_dim", nargs="?", default=2)
    parser.add_argument("--intervals", nargs="?", default=30)
    parser.add_argument("--total_time", nargs="?", default=0.2)
    parser.add_argument("--drift_param", nargs="?", default=0.1)
    parser.add_argument("--convergence_limit", nargs="?", default=0.999)
    args = parser.parse_args()

    #Initialize constants
    HILBERT_DIMENSION = int(args.hilbert_dim)
    NUM_OF_INTERVALS  = int(args.intervals)
    TOTAL_TIME        = float(args.total_time)
    DRIFT_PARAMETER   = float(args.drift_param)
    CONVERGENCE_LIMIT = float(args.convergence_limit)
    
    INITIAL_STATE = Statevector.from_int(0, dims=HILBERT_DIMENSION)
    TARGET_STATE  = random_statevector(HILBERT_DIMENSION, seed=0) #Set seed for reproducibility

    #Run Simple GRAPE algorithm
    simple_grape = SimpleGRAPE(HILBERT_DIMENSION, NUM_OF_INTERVALS, TOTAL_TIME, DRIFT_PARAMETER, CONVERGENCE_LIMIT, INITIAL_STATE, TARGET_STATE)

    (waveform_theta_x_coeffs, waveform_theta_y_coeffs) = simple_grape.run()
   
