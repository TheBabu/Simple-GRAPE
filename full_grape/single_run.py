#!/usr/bin/env python3

from qiskit.quantum_info import Statevector, random_unitary
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from full_grape import FullGRAPE
from graph_util.waveforms import generate_waveforms_plot
from graph_util.density import generate_density_plot

def main():
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hilbert_dim", nargs="?", type=int, default=2)
    parser.add_argument("--targets", nargs="?", type=int, default=2)
    parser.add_argument("--intervals", nargs="?", type=int, default=5)
    parser.add_argument("--total_time", nargs="?", type=float, default=5)
    parser.add_argument("--drift_param", nargs="?", type=float, default=1)
    parser.add_argument("--taylor_truncate_len", nargs="?", type=int, default=10)
    parser.add_argument("--init_seed", nargs="?", type=int, default=0)
    parser.add_argument("--target_states_seed", nargs="?", type=int, default=0)
    parser.add_argument("--no_plot_waveforms", action="store_true")
    parser.add_argument("--no_plot_density", action="store_true")
    parser.add_argument("--plot_density_time_interval", nargs="?", type=float, default=150)
    parser.add_argument("--check_grad", action="store_true")
    args = parser.parse_args()

    #Initialize constants
    HILBERT_DIMENSION          = args.hilbert_dim
    NUM_OF_TARGETS             = args.targets
    NUM_OF_INTERVALS           = args.intervals
    TOTAL_TIME                 = args.total_time
    DRIFT_PARAMETER            = args.drift_param
    TAYLOR_TRUNCATE_LEN        = args.taylor_truncate_len
    INIT_SEED                  = args.init_seed
    TARGET_STATES_SEED         = args.target_states_seed
    PLOT_WAVEFORMS             = not args.no_plot_waveforms
    PLOT_DENSITY               = not args.no_plot_density
    PLOT_DENSITY_TIME_INTERVAL = args.plot_density_time_interval
    CHECK_GRAD                 = args.check_grad

    INITIAL_STATES = [Statevector.from_int(i, dims=HILBERT_DIMENSION) for i in range(NUM_OF_TARGETS)]
    TARGET_STATES  = random_unitary(dims=HILBERT_DIMENSION, seed=TARGET_STATES_SEED).data[:NUM_OF_TARGETS] #Slice random unitary matrix columns to get random target states
    TARGET_STATES  = [Statevector(target_state) for target_state in TARGET_STATES]

    #Run Simple GRAPE algorithm
    simple_grape = FullGRAPE(HILBERT_DIMENSION,
                             NUM_OF_TARGETS,
                             NUM_OF_INTERVALS,
                             TOTAL_TIME,
                             DRIFT_PARAMETER,
                             TAYLOR_TRUNCATE_LEN,
                             INIT_SEED,
                             INITIAL_STATES,
                             TARGET_STATES,
                             CHECK_GRAD)

    if(CHECK_GRAD):
        grad_error = simple_grape.run()

        #DEBUG
        print(f"{grad_error=}")
        return

    (final_cost, theta_waveforms, unitary_list) = simple_grape.run()

    #DEBUG
    print(f"{final_cost=}")
    
    #Create data path
    data_path = Path(__file__).parents[1] / "data" / "single_run" /\
        f"full_grape_{HILBERT_DIMENSION}_dim_N_{NUM_OF_INTERVALS}_T_{TOTAL_TIME}_drift_param_{DRIFT_PARAMETER}"\
        f"_taylor_len_{TAYLOR_TRUNCATE_LEN}_target_state_seed_{TARGET_STATES_SEED}_init_seed_{INIT_SEED}"
    data_path.mkdir(parents=True, exist_ok=True)

    #Export data
    metadata_df = pd.DataFrame({
        "hilbert_dim"         : np.repeat(HILBERT_DIMENSION, NUM_OF_TARGETS),
        "num_of_targets"      : np.repeat(NUM_OF_TARGETS, NUM_OF_TARGETS),
        "num_of_intervals"    : np.repeat(NUM_OF_INTERVALS, NUM_OF_TARGETS),
        "total_time"          : np.repeat(TOTAL_TIME, NUM_OF_TARGETS),
        "drift_parameter"     : np.repeat(DRIFT_PARAMETER, NUM_OF_TARGETS),
        "taylor_truncate_len" : np.repeat(TAYLOR_TRUNCATE_LEN, NUM_OF_TARGETS),
        "init_seed"           : np.repeat(INIT_SEED, NUM_OF_TARGETS),
        "target_states_seed"  : np.repeat(TARGET_STATES_SEED, NUM_OF_TARGETS),
        "initial_states"      : INITIAL_STATES,
        "target_state"        : TARGET_STATES,
        "final_cost"          : np.repeat(final_cost, NUM_OF_TARGETS)
    })
    metadata_df.to_csv(data_path / "metadata.csv", index=False)

    theta_waveforms_df = pd.DataFrame({
        "theta": theta_waveforms,
    })
    theta_waveforms_df.to_csv(data_path / "theta_waveforms.csv", index=False)

    unitary_list_df = pd.DataFrame({
        "unitary": unitary_list,
    })
    unitary_list_df.to_csv(data_path / "unitary_list.csv", index=False)

    #Export waveforms plot if flagged true
    if(PLOT_WAVEFORMS):
        waveforms_plot = generate_waveforms_plot(theta_waveforms, final_cost, TOTAL_TIME, NUM_OF_INTERVALS)

        waveforms_plot.savefig(data_path / "waveforms_plot.png")
        waveforms_plot.savefig(data_path / "waveforms_plot.svg")

    #Export density animation if flagged true
    if(PLOT_DENSITY):
        for k, (initial_state, target_state) in enumerate(zip(INITIAL_STATES, TARGET_STATES)):
            density_animation = generate_density_plot(unitary_list, TOTAL_TIME, NUM_OF_INTERVALS, initial_state, target_state, HILBERT_DIMENSION, PLOT_DENSITY_TIME_INTERVAL)
            
            density_animation.save(data_path / f"density_animation_{k}.mp4")
            density_animation.save(data_path / f"density_animation_{k}.gif")

if __name__ == "__main__":
    main()

