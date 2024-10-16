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

    #Calculate real and imaginary target densities
    target_density_matrix = DensityMatrix(target_state)
    target_real_densities = np.real(target_density_matrix).ravel()
    target_imag_densities = np.imag(target_density_matrix)

    #Export animation
    data_path = Path(__file__).parents[1] / "data" / "density_plots" / Path(*FOLDER_PATH.parts[-2:])
    data_path.mkdir(parents=True, exist_ok=True)

    plt.rc("font",**{"family":"serif","serif":["Palatino"]})
    plt.rc("text", usetex=True)
    # plt.rcParams["text.latex.preamble"] = r"\DeclareUnicodeCharacter{03C1}{\ensuremath{\rho}}"

    density_figure = plt.figure(figsize=(15, 10))
    axis_real      = density_figure.add_subplot(221, projection="3d")
    axis_imag      = density_figure.add_subplot(222, projection="3d")
    axis_fidelity  = density_figure.add_subplot(212)

    def animate_plots(frame_num):
        #Clear subplots
        axis_real.cla()
        axis_imag.cla()
        axis_fidelity.cla()

        #Create real and imaginary densities
        current_density_matrix = DensityMatrix(state_list[frame_num])
        current_real_densities = np.real(current_density_matrix).ravel()
        current_imag_densities = np.imag(current_density_matrix).ravel()

        x_positions, y_positions = np.meshgrid(range(hilbert_dimension), range(hilbert_dimension))
        x_positions = x_positions.ravel()
        y_positions = y_positions.ravel()

        z_positions = [0] * len(current_real_densities)
        bar_widths  = [0.7] * len(current_real_densities)
        bar_depths  = [0.7] * len(current_real_densities)

        min_height    = -1
        max_height    = 1
        current_alpha = 0.8
        target_alpha  = 0.2

        #Plot real density matrix
        axis_real.set_title("Real Part", fontsize=20 * 1.15, pad=10)
        axis_real.set_zlim3d(min_height, max_height)
        axis_real.bar3d(x_positions, y_positions, z_positions, bar_widths, bar_depths, current_real_densities, alpha=current_alpha)
        axis_real.bar3d(x_positions, y_positions, z_positions, bar_widths, bar_depths, target_real_densities, alpha=target_alpha)

        #Plot imaginary density matrix
        axis_imag.set_title("Imaginary Part", fontsize=20 * 1.15, pad=10)
        axis_imag.set_zlim3d(min_height, max_height)
        axis_imag.bar3d(x_positions, y_positions, z_positions, bar_widths, bar_depths, current_imag_densities, alpha=current_alpha)
        axis_imag.bar3d(x_positions, y_positions, z_positions, bar_widths, bar_depths, target_real_densities, alpha=target_alpha)

        #Plot cost
        min_fidelity        = 0
        max_fidelity        = 1
        x_axis_fudge_factor = 0.3

        partial_fidelity_list = fidelity_list[:frame_num + 1]
        partial_time_points   = time_points[:frame_num + 1]

        axis_fidelity.set_title("Fidelity", fontsize=20 * 1.15, pad=10)
        axis_fidelity.set_xlabel("$\Omega t$", fontsize=20 * 1.15)
        axis_fidelity.set_ylabel("Fidelity", fontsize=20 * 1.15)
        axis_fidelity.set_xlim(0, time_points[-1] + x_axis_fudge_factor)
        axis_fidelity.set_ylim(min_fidelity, max_fidelity)
        axis_fidelity.tick_params(labelsize=15 * 1.15)
        axis_fidelity.plot(partial_time_points, partial_fidelity_list, "-o")

    density_animation = FuncAnimation(density_figure, animate_plots, interval=TIME_INTERVAL, frames=len(fidelity_list))
    density_animation.save(data_path / "density_animation.mp4")

