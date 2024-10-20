#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from qiskit.quantum_info import state_fidelity, Operator, Statevector, DensityMatrix #Ignore warning (Used in eval)

if __name__ == "__main__":
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=lambda p: Path(p).absolute())
    parser.add_argument("--time_interval", nargs="?", default=150)
    args = parser.parse_args()	

    #Initialize constants
    FOLDER_PATH   = args.folder_path
    TIME_INTERVAL = float(args.time_interval)

    unitary_list_df = pd.read_csv(FOLDER_PATH / "unitary_list.csv")
    metadata_df     = pd.read_csv(FOLDER_PATH / "metadata.csv")

    #Extract necessary metadata
    total_time        = float(metadata_df["total_time"][0])
    num_of_intervals  = int(metadata_df["num_of_intervals"][0])
    time_step         = total_time / num_of_intervals
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

    time_intervals = np.linspace(0, total_time, num_of_intervals + 1)
    
    #Calculate real and imaginary target densities
    target_density_matrix     = DensityMatrix(target_state)
    target_absolute_densities = np.abs(target_density_matrix).ravel()

    #Export animation
    num_of_params = 7
    data_path     = Path(__file__).parents[1] / "data" / "density_plots" / Path(*FOLDER_PATH.parts[-num_of_params:])
    data_path.mkdir(parents=True, exist_ok=True)

    plt.rc("font",**{"family":"serif","serif":["Palatino"]})
    plt.rc("text", usetex=True)
    plt.rcParams["text.latex.preamble"] = r"\usepackage{braket} \usepackage{amsmath}"

    density_figure      = plt.figure(figsize=(15, 10))
    axis_density        = density_figure.add_subplot(221, projection="3d")
    axis_density_target = density_figure.add_subplot(222, projection="3d")
    axis_fidelity       = density_figure.add_subplot(212)

    def animate_plots(frame_num):
        #Clear subplots
        axis_density.cla()
        axis_density_target.cla()
        axis_fidelity.cla()

        #Create real and imaginary densities
        current_density_matrix     = DensityMatrix(state_list[frame_num])
        absolute_current_densities = np.abs(current_density_matrix).ravel()

        x_positions, y_positions = np.meshgrid(range(hilbert_dimension), range(hilbert_dimension))
        x_positions = x_positions.ravel()
        y_positions = y_positions.ravel()

        z_positions = [0] * len(absolute_current_densities)
        bar_widths  = [0.7] * len(absolute_current_densities)
        bar_depths  = [0.7] * len(absolute_current_densities)

        min_height = 0
        max_height = 1
        bar_alpha  = 0.8

        #Plot density matrix
        axis_density.set_title("Absolute Value of Density Matrix\n" r"($\lvert\ket{\psi}\bra{\psi}\rvert$)", fontsize=20 * 1.15, pad=10)
        axis_density.set_zlim3d(min_height, max_height)
        axis_density.bar3d(x_positions, y_positions, z_positions, bar_widths, bar_depths, absolute_current_densities, alpha=bar_alpha)

        #Plot target density matrix
        axis_density_target.set_title("Absolute Value of Target Density Matrix\n" r"($\lvert\ket{\psi_t}\bra{\psi_t}\rvert$)", fontsize=20 * 1.15, pad=10)
        axis_density_target.set_zlim3d(min_height, max_height)
        axis_density_target.bar3d(x_positions, y_positions, z_positions, bar_widths, bar_depths, target_absolute_densities, alpha=bar_alpha)

        #Plot cost
        min_fidelity        = 0
        max_fidelity        = 1

        partial_fidelity_list = fidelity_list[:frame_num + 1]
        partial_time_points   = time_intervals[:frame_num + 1]

        axis_fidelity.set_title("Fidelity", fontsize=20 * 1.15, pad=10)
        axis_fidelity.set_xlabel("$\Omega t$", fontsize=20 * 1.15)
        axis_fidelity.set_ylabel("Fidelity", fontsize=20 * 1.15)
        axis_fidelity.set_xlim(0, time_intervals[-1])
        axis_fidelity.set_ylim(min_fidelity, max_fidelity)
        axis_fidelity.tick_params(labelsize=15 * 1.15)
        axis_fidelity.plot(partial_time_points, partial_fidelity_list, "-o")


    density_animation = FuncAnimation(density_figure, animate_plots, interval=TIME_INTERVAL, frames=len(fidelity_list))
    density_animation.save(data_path / "density_animation.mp4")
    density_animation.save(data_path / "density_animation.gif")

    #DEBUG
    # plt.show()

