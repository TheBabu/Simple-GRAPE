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
    parser.add_argument("--intervals", nargs="?", default=5)
    parser.add_argument("--total_time", nargs="?", default=3)
    parser.add_argument("--drift_param", nargs="?", default=1)
    parser.add_argument("--taylor_truncate_len", nargs="?", default=10)
    parser.add_argument("--init_seed", nargs="?", default=0)
    parser.add_argument("--target_state_seed", nargs="?", default=0)
    args = parser.parse_args()

    #Initialize constants
    HILBERT_DIMENSION   = int(args.hilbert_dim)
    NUM_OF_INTERVALS    = int(args.intervals)
    TOTAL_TIME          = float(args.total_time)
    DRIFT_PARAMETER     = float(args.drift_param)
    TAYLOR_TRUNCATE_LEN = int(args.taylor_truncate_len)
    INIT_SEED           = int(args.init_seed)
    TARGET_STATE_SEED   = int(args.target_state_seed)
    
    INITIAL_STATE = Statevector.from_int(0, dims=HILBERT_DIMENSION)
    TARGET_STATE  = random_statevector(HILBERT_DIMENSION, seed=TARGET_STATE_SEED) #Set seed for reproducibility

    #Run Simple GRAPE algorithm
    simple_grape = SimpleGRAPE(HILBERT_DIMENSION, NUM_OF_INTERVALS, TOTAL_TIME, DRIFT_PARAMETER, TAYLOR_TRUNCATE_LEN, INIT_SEED, INITIAL_STATE, TARGET_STATE)

    (final_cost, theta_x_waveforms, theta_y_waveforms, unitary_list) = simple_grape.run()
    
    #Create data path
    data_path   = Path(__file__).parents[1] / "data" / "grape_data" /\
        f"{HILBERT_DIMENSION}_dim" / f"N_{NUM_OF_INTERVALS}" / f"T_{TOTAL_TIME}" / f"drift_param_{DRIFT_PARAMETER}" /\
        f"taylor_len_{TAYLOR_TRUNCATE_LEN}" / f"target_state_seed_{TARGET_STATE_SEED}" / f"init_seed_{INIT_SEED}"
    data_path.mkdir(parents=True, exist_ok=True)

    #Export data
    metadata_df = pd.DataFrame({
        "hilbert_dim"         : [HILBERT_DIMENSION],
        "num_of_intervals"    : [NUM_OF_INTERVALS],
        "total_time"          : [TOTAL_TIME],
        "drift_parameter"     : [DRIFT_PARAMETER],
        "taylor_truncate_len" : [TAYLOR_TRUNCATE_LEN],
        "init_seed"           : [INIT_SEED],
        "target_state_seed"   : [TARGET_STATE_SEED],
        "initial_state"       : [INITIAL_STATE],
        "target_state"        : [TARGET_STATE],
        "final_cost"          : [final_cost]
    })
    metadata_df.to_csv(data_path / "metadata.csv")

    theta_waveforms_df = pd.DataFrame({
        "theta_x": theta_x_waveforms,
        "theta_y": theta_y_waveforms
    })
    theta_waveforms_df.to_csv(data_path / "theta_waveforms.csv")

    unitary_list_df = pd.DataFrame({
        "unitary": unitary_list,
    })
    unitary_list_df.to_csv(data_path / "unitary_list.csv")
   
