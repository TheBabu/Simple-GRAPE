#!/usr/bin/env python3

from qiskit.quantum_info import Statevector, random_statevector
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from simple_grape import SimpleGRAPE
from graph_util.waveforms import generate_waveforms_plot
from graph_util.density import generate_density_plot

def main():
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hilbert_dim", nargs="?", default=2)
    parser.add_argument("--intervals", nargs="?", default=5)
    parser.add_argument("--total_time", nargs="?", default=3)
    parser.add_argument("--drift_param", nargs="?", default=1)
    parser.add_argument("--taylor_truncate_len", nargs="?", default=10)
    parser.add_argument("--init_seed", nargs="?", default=0)
    parser.add_argument("--target_state_seed", nargs="?", default=0)
    parser.add_argument("--plot_waveforms", action="store_false")
    parser.add_argument("--plot_density", action="store_false")
    parser.add_argument("--plot_density_time_interval", nargs="?", default=150)
    parser.add_argument("--check_grad", action="store_true")
    args = parser.parse_args()

    #Initialize constants
    HILBERT_DIMENSION          = int(args.hilbert_dim)
    NUM_OF_INTERVALS           = int(args.intervals)
    TOTAL_TIME                 = float(args.total_time)
    DRIFT_PARAMETER            = float(args.drift_param)
    TAYLOR_TRUNCATE_LEN        = int(args.taylor_truncate_len)
    INIT_SEED                  = int(args.init_seed)
    TARGET_STATE_SEED          = int(args.target_state_seed)
    PLOT_WAVEFORMS             = args.plot_waveforms
    PLOT_DENSITY               = args.plot_density
    PLOT_DENSITY_TIME_INTERVAL = int(args.plot_density_time_interval)
    CHECK_GRAD                 = args.check_grad
    
    INITIAL_STATE = Statevector.from_int(0, dims=HILBERT_DIMENSION)
    TARGET_STATE  = random_statevector(HILBERT_DIMENSION, seed=TARGET_STATE_SEED) #Set seed for reproducibility

    #Run Simple GRAPE algorithm
    simple_grape = SimpleGRAPE(HILBERT_DIMENSION,
                               NUM_OF_INTERVALS,
                               TOTAL_TIME,
                               DRIFT_PARAMETER,
                               TAYLOR_TRUNCATE_LEN,
                               INIT_SEED,
                               INITIAL_STATE,
                               TARGET_STATE,
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
        f"{HILBERT_DIMENSION}_dim_N_{NUM_OF_INTERVALS}_T_{TOTAL_TIME}_drift_param_{DRIFT_PARAMETER}"\
        f"_taylor_len_{TAYLOR_TRUNCATE_LEN}_target_state_seed_{TARGET_STATE_SEED}_init_seed_{INIT_SEED}"
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
        theta_waveforms = np.insert(theta_waveforms, 0, theta_waveforms[0]) #Prepend first initial data point, so first interval is shown
        
        waveforms_plot = generate_waveforms_plot(theta_waveforms, final_cost, TOTAL_TIME, NUM_OF_INTERVALS)

        waveforms_plot.savefig(data_path / "waveforms.png")
        waveforms_plot.savefig(data_path / "waveforms.svg")

    #Export density animation if flagged true
    if(PLOT_DENSITY):
        density_animation = generate_density_plot(unitary_list, TOTAL_TIME, NUM_OF_INTERVALS, INITIAL_STATE, TARGET_STATE, HILBERT_DIMENSION, PLOT_DENSITY_TIME_INTERVAL)
        
        density_animation.save(data_path / "density_animation.mp4")
        density_animation.save(data_path / "density_animation.gif")

if __name__ == "__main__":
    main()

