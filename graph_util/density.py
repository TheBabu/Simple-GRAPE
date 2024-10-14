#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from qiskit.visualization import plot_state_city
from qiskit.quantum_info import Statevector

if __name__ == "__main__":
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=lambda p: Path(p).absolute())
    parser.add_argument("--time_interval", nargs="?", default=150)
    parser.add_argument("--alpha", nargs="?", default=0.9)
    args = parser.parse_args()	

    #Initialize constants
    FOLDER_PATH   = args.folder_path
    TIME_INTERVAL = float(args.time_interval)
    ALPHA         = float(args.alpha)

    optimization_df = pd.read_csv(FOLDER_PATH / "optimization_data.csv")

    #Extract optimization data
    cost_list = optimization_df["cost"]
    states    = optimization_df["state"]

    #Export animation
    data_path = Path(__file__).parents[1] / "data" / "density_plots" / Path(*FOLDER_PATH.parts[-2:])
    data_path.mkdir(parents=True, exist_ok=True)

    # plt.rc("font",**{"family":"serif","serif":["Palatino"]})
    # plt.rc("text", usetex=True)

    def animate(i):
        #Data to plot
        partial_cost_list = cost_list[:i + 1]
        state             = eval(states[i])

        #Clear previous plot
        plt.cla()

        #Plot density matrix
        plot_state_city(state)

        #Plot cost
        # title_str = "Cost"
        # plt.title(title_str, fontsize=20 * 1.15, pad=10)
        # plt.plot(partial_cost_list, "-o")
        # plt.xlabel("Iteration", fontsize=20 * 1.15)
        # plt.ylabel("Cost", fontsize=20 * 1.15)
        # plt.xlim(0, len(cost_list))
        # plt.ylim(-1, 0)
        # plt.tick_params(labelsize=15 * 1.15)

    fig = plt.figure(figsize=(15, 9), linewidth=2 * 1.15)
    ani = FuncAnimation(fig, animate, interval=TIME_INTERVAL)
    plt.show()

