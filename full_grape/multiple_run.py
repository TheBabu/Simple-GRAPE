#!/usr/bin/env python3

from qiskit.quantum_info import Statevector, random_unitary
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing.pool import Pool

from full_grape import FullGRAPE
from graph_util.colormap import reduce_grape_data

#HACK: To allow different functions in multithreading
def smap(f, *args):
    return f(*args)

def main():
    #Parse system arguments
    parser       = argparse.ArgumentParser()
    parser_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument("--hilbert_dim", nargs="?", type=int, default=2)
    parser.add_argument("--targets", nargs="?", type=int, default=2)
    parser.add_argument("--num_of_intervals_interval", nargs=3, metavar=("(START", "END", "STEP)"), type=int, default=[1, 7, 1])
    parser.add_argument("--total_time_interval", nargs=3, metavar=("(START", "END", "STEP)"), type=float, default=[3.0, 7.0, 0.5])
    parser.add_argument("--drift_param_interval", nargs=3, metavar=("(START", "END", "STEP)"), type=float, default=[1.0, 1.0, 1])
    parser.add_argument("--taylor_truncate_len_interval", nargs=3, metavar=("(START", "END", "STEP)"), type=int, default=[10, 10, 1])
    parser.add_argument("--init_seed_interval", nargs=3, metavar=("(START", "END", "STEP)"), type=int, default=[0, 9, 1])
    parser.add_argument("--target_states_seed_interval", nargs=3, metavar=("(START", "END", "STEP)"), type=int, default=[0, 9, 1])
    parser_group.add_argument("--check_grad", action="store_true")
    parser_group.add_argument("--folder_name")
    parser.add_argument("--no_reduce_grape_data", action="store_true")
    args = parser.parse_args()

    #Initialize constants
    HILBERT_DIMENSION            = args.hilbert_dim
    NUM_OF_TARGETS               = args.targets
    NUM_OF_INTERVALS_INTERVAL    = args.num_of_intervals_interval
    TOTAL_TIME_INTERVAL          = args.total_time_interval
    DRIFT_PARAM_INTERVAL         = args.drift_param_interval
    TAYLOR_TRUNCATE_LEN_INTERVAL = args.taylor_truncate_len_interval
    INIT_SEED_INTERVAL           = args.init_seed_interval
    TARGET_STATES_SEED_INTERVAL  = args.target_states_seed_interval
    CHECK_GRAD                   = args.check_grad
    FOLDER_NAME                  = args.folder_name
    REDUCE_GRAPE_DATA            = not args.no_reduce_grape_data
    
    INITIAL_STATES = [Statevector.from_int(i, dims=HILBERT_DIMENSION) for i in range(NUM_OF_TARGETS)]

    grape_data = []
    pool       = Pool()
    #TODO: Fix edge case where interval is not evenly divided by step size?
    for num_of_intervals in np.arange(NUM_OF_INTERVALS_INTERVAL[0], NUM_OF_INTERVALS_INTERVAL[1] + NUM_OF_INTERVALS_INTERVAL[2], NUM_OF_INTERVALS_INTERVAL[2]):
        for total_time in np.arange(TOTAL_TIME_INTERVAL[0], TOTAL_TIME_INTERVAL[1] + TOTAL_TIME_INTERVAL[2], TOTAL_TIME_INTERVAL[2]):
            for drift_parameter in np.arange(DRIFT_PARAM_INTERVAL[0], DRIFT_PARAM_INTERVAL[1] + DRIFT_PARAM_INTERVAL[2], DRIFT_PARAM_INTERVAL[2]):
                for taylor_truncate_len in np.arange(TAYLOR_TRUNCATE_LEN_INTERVAL[0], TAYLOR_TRUNCATE_LEN_INTERVAL[1] + TAYLOR_TRUNCATE_LEN_INTERVAL[2], TAYLOR_TRUNCATE_LEN_INTERVAL[2]):
                    for target_states_seed in np.arange(TARGET_STATES_SEED_INTERVAL[0], TARGET_STATES_SEED_INTERVAL[1] + TARGET_STATES_SEED_INTERVAL[2], TARGET_STATES_SEED_INTERVAL[2]):
                        target_states = random_unitary(dims=HILBERT_DIMENSION, seed=target_states_seed).data[:NUM_OF_TARGETS] #Slice random unitary matrix columns to get random target states
                        target_states = [Statevector(target_state) for target_state in target_states]

                        simple_grape_list = [
                            FullGRAPE(HILBERT_DIMENSION,
                                      NUM_OF_TARGETS,
                                      num_of_intervals,
                                      total_time,
                                      drift_parameter,
                                      taylor_truncate_len,
                                      init_seed,
                                      INITIAL_STATES,
                                      target_states,
                                      CHECK_GRAD)
                            for init_seed in np.arange(INIT_SEED_INTERVAL[0], INIT_SEED_INTERVAL[1] + INIT_SEED_INTERVAL[2], INIT_SEED_INTERVAL[2])
                        ]
                        simple_grape_run_list = [simple_grape.run for simple_grape in simple_grape_list]

                        #Multithread each initial waveform seed
                        pool_result = pool.map(smap, simple_grape_run_list)
                    
                        if(CHECK_GRAD):
                            for init_seed, grad_error in enumerate(pool_result):
                                #DEBUG
                                print(f"{num_of_intervals=:3}, {total_time=:20.17f}, {drift_parameter=:5.3f}, {taylor_truncate_len=:3}, {target_states_seed=:3}, {init_seed=:3}, {grad_error=:20.17f}") 

                            #DEBUG
                            print()

                            continue

                        #Append final cost data to grape_data
                        for init_seed, (final_cost, _, _) in enumerate(pool_result):
                            grape_data.append((num_of_intervals, total_time, drift_parameter, taylor_truncate_len, target_states_seed, init_seed, final_cost))
                            
                            #DEBUG
                            print(f"{num_of_intervals=:3}, {total_time=:20.17f}, {drift_parameter=:5.3f}, {taylor_truncate_len=:3}, {target_states_seed=:3}, {init_seed=:3}, {final_cost=:20.17f}")

                        #DEBUG
                        print()

    #Close mulithreading pool
    pool.close()

    #Quit if checking gradients
    if(CHECK_GRAD):
        return

    #Create data path
    data_path = Path(__file__).parents[1] / "data" / "multiple_run" / FOLDER_NAME
    data_path.mkdir(parents=True, exist_ok=True)

    #Export data
    num_of_intervals_list,\
    total_time_list,\
    drift_parameter_list,\
    taylor_truncate_len_list,\
    target_state_seed_list,\
    init_seed_list,\
    final_cost_list = np.array(grape_data).T

    metadata_df = pd.DataFrame({
        "hilbert_dim"                  : [HILBERT_DIMENSION] * 2 + [0], #Need to match length of other intervals
        "num_of_targets"               : [NUM_OF_TARGETS] * 2 + [0], #Need to match length of other intervals
        "num_of_intervals_interval"    : NUM_OF_INTERVALS_INTERVAL,
        "total_time_interval"          : TOTAL_TIME_INTERVAL,
        "drift_param_interval"         : DRIFT_PARAM_INTERVAL,
        "taylor_truncate_len_interval" : TAYLOR_TRUNCATE_LEN_INTERVAL,
        "init_seed_interval"           : INIT_SEED_INTERVAL,
        "target_states_seed_interval"  : TARGET_STATES_SEED_INTERVAL
    }, index=(["start", "end", "step"]))
    metadata_df.to_csv(data_path / "metadata.csv")

    grape_data_df = pd.DataFrame({
        "num_of_intervals"    : num_of_intervals_list,
        "total_time"          : total_time_list,
        "drift_parameter"     : drift_parameter_list,
        "taylor_truncate_len" : taylor_truncate_len_list,
        "init_seed"           : init_seed_list,
        "target_state_seed"   : target_state_seed_list,
        "final_cost"          : final_cost_list
    })
    grape_data_df.to_csv(data_path / "grape_data.csv", index=False)

    #Reduce grape_data if flagged true
    if(REDUCE_GRAPE_DATA):
        reduced_grape_data_df = reduce_grape_data(grape_data_df)

        reduced_grape_data_df.to_csv(data_path / "reduced_grape_data.csv", index=False)

if __name__ == "__main__":
    main()

