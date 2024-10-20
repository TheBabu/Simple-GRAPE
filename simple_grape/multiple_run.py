#!/usr/bin/env python3

from qiskit.quantum_info import Statevector, random_statevector
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from simple_grape import SimpleGRAPE

if __name__ == "__main__":
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hilbert_dim", nargs="?", default=2)
    parser.add_argument("--interval_min", nargs="?", default=1)
    parser.add_argument("--interval_max", nargs="?", default=10)
    parser.add_argument("--interval_num_steps", nargs="?", default=100)
    parser.add_argument("--total_time_min", nargs="?", default=0)
    parser.add_argument("--total_time_max", nargs="?", default=10)
    parser.add_argument("--total_time_num_steps", nargs="?", default=100)
    parser.add_argument("--drift_param", nargs="?", default=1)
    parser.add_argument("--taylor_truncate_len", nargs="?", default=10)
    parser.add_argument("--num_init_seeds", nargs="?", default=10)
    parser.add_argument("--num_target_state_seeds", nargs="?", default=10)
    args = parser.parse_args()

    #Initialize constants
    HILBERT_DIMENSION      = int(args.hilbert_dim)
    INTERVAL_MIN           = int(args.interval_min)
    INTERVAL_MAX           = int(args.interval_max)
    TOTAL_TIME_MIN         = float(args.total_time_min)
    TOTAL_TIME_MAX         = float(args.total_time_max)
    TOTAL_TIME_NUM_STEPS   = int(args.total_time_num_steps)
    DRIFT_PARAMETER        = float(args.drift_param)
    TAYLOR_TRUNCATE_LEN    = int(args.taylor_truncate_len)
    NUM_INIT_SEEDS         = int(args.num_init_seeds)
    NUM_TARGET_STATE_SEEDS = int(args.num_target_state_seeds)
    
    INITIAL_STATE = Statevector.from_int(0, dims=HILBERT_DIMENSION)

    interval_list   = np.array(range(INTERVAL_MIN, INTERVAL_MAX + 1))
    total_time_list = np.linspace(TOTAL_TIME_MIN, TOTAL_TIME_MAX, TOTAL_TIME_NUM_STEPS)

    for num_of_intervals in interval_list:
        for total_time in total_time_list:
            for target_state_seed in range(NUM_TARGET_STATE_SEEDS):
                target_state  = random_statevector(HILBERT_DIMENSION, seed=target_state_seed) #Set seed for reproducibility

                #TODO: Multithread
                for init_seed in range(NUM_INIT_SEEDS):
                    #Run Simple GRAPE algorithm
                    simple_grape = SimpleGRAPE(HILBERT_DIMENSION,
                                               num_of_intervals,
                                               total_time,
                                               DRIFT_PARAMETER,
                                               TAYLOR_TRUNCATE_LEN,
                                               init_seed,
                                               INITIAL_STATE,
                                               target_state)

                    (final_cost, theta_x_waveforms, theta_y_waveforms, unitary_list) = simple_grape.run()
                    
                    #DEBUG
                    print(f"{num_of_intervals=:3}, {total_time=:20.17f}, {target_state_seed=:3}, {init_seed=:3}, {final_cost=:20.17f}") 
                    
                    #Create data path
                    data_path = Path(__file__).parents[1] / "data" / "grape_data" /\
                        f"{HILBERT_DIMENSION}_dim" / f"N_{num_of_intervals}" / f"T_{total_time}" / f"drift_param_{DRIFT_PARAMETER}" /\
                        f"taylor_len_{TAYLOR_TRUNCATE_LEN}" / f"target_state_seed_{target_state_seed}" / f"init_seed_{init_seed}"
                    data_path.mkdir(parents=True, exist_ok=True)

                    #Export data
                    metadata_df = pd.DataFrame({
                        "hilbert_dim"         : [HILBERT_DIMENSION],
                        "num_of_intervals"    : [num_of_intervals],
                        "total_time"          : [total_time],
                        "drift_parameter"     : [DRIFT_PARAMETER],
                        "taylor_truncate_len" : [TAYLOR_TRUNCATE_LEN],
                        "init_seed"           : [init_seed],
                        "target_state_seed"   : [target_state_seed],
                        "initial_state"       : [INITIAL_STATE],
                        "target_state"        : [target_state],
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
                
                #DEBUG
                print()

