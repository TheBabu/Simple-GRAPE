#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from fractions import Fraction
from qiskit.quantum_info import state_fidelity, Operator, Statevector, DensityMatrix #Ignore warning (Used in eval)

def generate_density_plot(unitary_list, total_time, num_of_intervals, initial_state, target_state, hilbert_dimension, time_interval):
    unitary_list  = [Operator(np.eye(hilbert_dimension))] + unitary_list #Prepend identity operator for time = 0
    fidelity_list = []
    state_list    = []
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
    target_absolute_densities = np.abs(target_density_matrix).ravel(order="F")

    #Generate labels for density matrix
    spin             = Fraction((hilbert_dimension - 1) / 2)
    spin_eigenvalues = np.arange(spin, -spin - 1, -1)
    x_labels         = [r"$\ket{" f"{spin_eigenvalue}" r"}$" for spin_eigenvalue in spin_eigenvalues]
    y_labels         = [r"$\bra{" f"{spin_eigenvalue}" r"}$" for spin_eigenvalue in spin_eigenvalues]
    
    #TODO: Make not global?
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
        absolute_current_densities = np.abs(current_density_matrix).ravel(order="F")

        x_positions, y_positions = np.meshgrid(range(hilbert_dimension), range(hilbert_dimension))
        x_positions = x_positions.ravel()
        y_positions = y_positions.ravel()

        bar_length  = 0.7
        z_positions = [0] * len(absolute_current_densities)
        bar_widths  = [bar_length] * len(absolute_current_densities)
        bar_depths  = [bar_length] * len(absolute_current_densities)

        min_height     = 0
        max_height     = 1
        bar_alpha      = 0.8
        tick_shift     = bar_length / 2
        tick_intervals = np.linspace(tick_shift + min(x_positions), tick_shift + max(y_positions), hilbert_dimension)

        #Plot density matrix
        axis_density.set_title("Absolute Value of Density Matrix\n" r"($\lvert\ket{\psi}\bra{\psi}\rvert$)", fontsize=20 * 1.15, pad=10)
        axis_density.set_zlim3d(min_height, max_height)
        axis_density.bar3d(x_positions, y_positions, z_positions, bar_widths, bar_depths, absolute_current_densities, alpha=bar_alpha)
        axis_density.set_xticks(tick_intervals, x_labels)
        axis_density.set_yticks(tick_intervals, y_labels)

        #Plot target density matrix
        axis_density_target.set_title("Absolute Value of Target Density Matrix\n" r"($\lvert\ket{\psi_t}\bra{\psi_t}\rvert$)", fontsize=20 * 1.15, pad=10)
        axis_density_target.set_zlim3d(min_height, max_height)
        axis_density_target.bar3d(x_positions, y_positions, z_positions, bar_widths, bar_depths, target_absolute_densities, alpha=bar_alpha)
        axis_density_target.set_xticks(tick_intervals, x_labels)
        axis_density_target.set_yticks(tick_intervals, y_labels)

        #Plot cost
        min_fidelity = 0
        max_fidelity = 1

        partial_fidelity_list = fidelity_list[:frame_num + 1]
        partial_time_points   = time_intervals[:frame_num + 1]

        axis_fidelity.set_title("Fidelity", fontsize=20 * 1.15, pad=10)
        axis_fidelity.set_xlabel("$\Omega t$", fontsize=20 * 1.15)
        axis_fidelity.set_ylabel("Fidelity", fontsize=20 * 1.15)
        axis_fidelity.set_xlim(0, time_intervals[-1])
        axis_fidelity.set_ylim(min_fidelity, max_fidelity)
        axis_fidelity.tick_params(labelsize=15 * 1.15)
        axis_fidelity.plot(partial_time_points, partial_fidelity_list, "-o")

    density_animation = FuncAnimation(density_figure, animate_plots, interval=time_interval, frames=len(fidelity_list))

    return density_animation

def main():
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=lambda p: Path(p).absolute())
    parser.add_argument("--time_interval", nargs="?", type=float, default=150)
    args = parser.parse_args()	

    #Initialize constants
    FOLDER_PATH   = args.folder_path
    TIME_INTERVAL = args.time_interval

    #Read raw data
    try:
        metadata_df = pd.read_csv(FOLDER_PATH / "metadata.csv")
    except FileNotFoundError as exception:
        raise Exception("Cannot read metadata.csv in given path") from exception
    
    try:
        unitary_list_df = pd.read_csv(FOLDER_PATH / "unitary_list.csv")
    except FileNotFoundError as exception:
        raise Exception("Cannot read unitary_list.csv (Only single_run.py produces unitary list data)") from exception

    #Extract necessary metadata
    total_time        = float(metadata_df["total_time"][0])
    num_of_intervals  = int(metadata_df["num_of_intervals"][0])
    initial_state     = eval(metadata_df["initial_state"][0])
    target_state      = eval(metadata_df["target_state"][0])
    hilbert_dimension = int(metadata_df["hilbert_dim"][0])

    #Extract unitary_list data
    unitary_list = [eval(unitary) for unitary in unitary_list_df["unitary"]]

    #Generate density animation
    density_animation = generate_density_plot(unitary_list, total_time, num_of_intervals, initial_state, target_state, hilbert_dimension, TIME_INTERVAL)
    
    #Export animation
    density_animation.save(FOLDER_PATH / "density_animation.mp4")
    density_animation.save(FOLDER_PATH / "density_animation.gif")

if __name__ == "__main__":
    main()

