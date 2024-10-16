#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from qiskit.quantum_info import state_fidelity, Operator, Statevector #Ignore warning (Used in eval)
from qiskit.visualization import plot_state_city

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

    unitary_list_df = pd.read_csv(FOLDER_PATH / "unitary_list.csv")
    metadata_df     = pd.read_csv(FOLDER_PATH / "metadata.csv")

    #Extract necessary metadata
    total_time        = float(metadata_df["total_time"][0])
    num_of_intervals  = int(metadata_df["num_of_intervals"][0])
    time_step         = total_time / (num_of_intervals - 1)
    initial_state     = eval(metadata_df["initial_state"][0])
    target_state      = eval(metadata_df["target_state"][0])
    hilbert_dimension = int(metadata_df["hilbert_dim"][0])

    #Extract unitary_list data
    unitary_list = [Operator(np.eye(hilbert_dimension))] + [eval(unitary) for unitary in unitary_list_df["unitary"]]

    fidelity_list         = []
    state_list            = []
    current_total_unitary = Operator(np.eye(hilbert_dimension))
    for current_unitary in unitary_list:
        current_total_unitary = current_total_unitary.compose(current_unitary)

        current_evolved_state = initial_state.evolve(current_total_unitary)
        current_fidelity      = state_fidelity(current_evolved_state, target_state)

        fidelity_list.append(current_fidelity)
        state_list.append(current_evolved_state)

    time_points = np.arange(0, total_time + 2 * time_step, time_step)

    #Export animation
    data_path = Path(__file__).parents[1] / "data" / "density_plots" / Path(*FOLDER_PATH.parts[-2:])
    data_path.mkdir(parents=True, exist_ok=True)

    plt.rc("font",**{"family":"serif","serif":["Palatino"]})
    plt.rc("text", usetex=True)
    plt.rcParams["text.latex.preamble"] = r"\DeclareUnicodeCharacter{03C1}{\ensuremath{\rho}}"

    density_figure = plt.figure(figsize=(15, 10))
    axis_real      = density_figure.add_subplot(221, projection="3d")
    axis_imag      = density_figure.add_subplot(222, projection="3d")
    axis_fidelity  = density_figure.add_subplot(212)

    min_fidelity        = 0
    max_fidelity        = 1
    x_axis_fudge_factor = 0.3

    def animate_plots(frame_num):
        #Clear subplots
        axis_real.cla()
        axis_imag.cla()
        axis_fidelity.cla()

        #Plot density matrix
        current_state = state_list[frame_num]

        plot_state_city(current_state, ax_real=axis_real, ax_imag=axis_imag, alpha=ALPHA)
        axis_real.set_zlim3d(-1, 1)
        axis_imag.set_zlim3d(-1, 1)

        #Plot cost
        partial_fidelity_list = fidelity_list[:frame_num + 1]
        partial_time_points   = time_points[:frame_num + 1]

        axis_fidelity.set_title("Fidelity", fontsize=20 * 1.15, pad=10)
        axis_fidelity.set_xlabel("$\Omega t$", fontsize=20 * 1.15)
        axis_fidelity.set_ylabel("Fidelity", fontsize=20 * 1.15)
        axis_fidelity.set_xlim(0, len(fidelity_list) + x_axis_fudge_factor)
        axis_fidelity.set_ylim(min_fidelity, max_fidelity)
        axis_fidelity.tick_params(labelsize=15 * 1.15)
        axis_fidelity.plot(partial_time_points, partial_fidelity_list, "-o")

    density_animation = FuncAnimation(density_figure, animate_plots, interval=TIME_INTERVAL, frames=len(fidelity_list))
    plt.show()

