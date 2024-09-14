#!/usr/bin/env python3

from qiskit.quantum_info import Statevector, random_statevector
import argparse

from simple_grape import SimpleGRAPE

if __name__ == "__main__":
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hilbert_dim", nargs="?", default=2)
    parser.add_argument("--intervals", nargs="?", default=100)
    parser.add_argument("--total_time", nargs="?", default=3)
    parser.add_argument("--drift_param", nargs="?", default=3)
    args = parser.parse_args()

    #Initialize constants
    HILBERT_DIMENSION = int(args.hilbert_dim)
    NUM_OF_INTERVALS  = int(args.intervals)
    TOTAL_TIME        = float(args.total_time)
    DRIFT_PARAMETER   = float(args.drift_param)
    
    INITIAL_STATE = Statevector.from_int(0, dims=HILBERT_DIMENSION)
    TARGET_STATE  = random_statevector(HILBERT_DIMENSION, seed=0) #Set seed

    #Run Simple GRAPE algorithm
    simple_grape = SimpleGRAPE(HILBERT_DIMENSION, NUM_OF_INTERVALS, TOTAL_TIME, DRIFT_PARAMETER, INITIAL_STATE, TARGET_STATE)

    fidelity_list = simple_grape.run()
   
