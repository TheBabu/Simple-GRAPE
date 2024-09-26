#!/usr/bin/env python3

from qiskit.quantum_info import Statevector, random_statevector
import argparse
from pathlib import Path
import pandas as pd

from simple_grape import SimpleGRAPE

if __name__ == "__main__":
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hilbert_dim", nargs="?", default=2)
    parser.add_argument("--intervals", nargs="?", default=250)
    parser.add_argument("--total_time", nargs="?", default=150)
    parser.add_argument("--drift_param", nargs="?", default=1)
    parser.add_argument("--init_seed", nargs="?", default=0)
    parser.add_argument("--target_state_seed", nargs="?", default=0)
    args = parser.parse_args()

    #Initialize constants
    HILBERT_DIMENSION = int(args.hilbert_dim)
    NUM_OF_INTERVALS  = int(args.intervals)
    TOTAL_TIME        = float(args.total_time)
    DRIFT_PARAMETER   = float(args.drift_param)
    INIT_SEED         = int(args.init_seed)
    TARGET_STATE_SEED = int(args.target_state_seed)
    
    INITIAL_STATE = Statevector.from_int(0, dims=HILBERT_DIMENSION)
    TARGET_STATE  = random_statevector(HILBERT_DIMENSION, seed=TARGET_STATE_SEED) #Set seed for reproducibility

    #Run Simple GRAPE algorithm
    simple_grape = SimpleGRAPE(HILBERT_DIMENSION, NUM_OF_INTERVALS, TOTAL_TIME, DRIFT_PARAMETER, INIT_SEED, INITIAL_STATE, TARGET_STATE)

    (final_cost, theta_x_waveforms, theta_y_waveforms) = simple_grape.run()
    
    #Create data path
    folder_name = f"N_{NUM_OF_INTERVALS}_T_{TOTAL_TIME}_drift_param_{DRIFT_PARAMETER}_seed_{INIT_SEED}_target_state_seed_{TARGET_STATE_SEED}"
    data_path   = Path(__file__).parents[1] / "data" / "grape_data" / f"{HILBERT_DIMENSION}_dim" / folder_name
    data_path.mkdir(parents=True, exist_ok=True)

    #Export data
    metadata_df = pd.DataFrame({
        "hilbert_dim"       : [HILBERT_DIMENSION],
        "num_of_intervals"  : [NUM_OF_INTERVALS],
        "total_time"        : [TOTAL_TIME],
        "drift_parameter"   : [DRIFT_PARAMETER],
        "init_seed"         : [INIT_SEED],
        "target_state_seed" : [TARGET_STATE_SEED],
        "initial_state"     : [INITIAL_STATE],
        "target_state"      : [TARGET_STATE],
        "final_cost"        : [final_cost]
    })
    metadata_df.to_csv(data_path / "metadata.csv")

    theta_waveforms_df = pd.DataFrame({
        "theta_x": theta_x_waveforms,
        "theta_y": theta_y_waveforms
    })
    theta_waveforms_df.to_csv(data_path / "theta_waveforms.csv")
   
