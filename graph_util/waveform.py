#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=lambda p: Path(p).absolute())
    args = parser.parse_args()	

    #Initialize constants
    FOLDER_PATH = args.folder_path

    metadata_df        = pd.read_csv(FOLDER_PATH / "metadata.csv")
    theta_waveforms_df = pd.read_csv(FOLDER_PATH / "theta_waveforms.csv")

    #Extract necessary metadata
    final_cost       = metadata_df["final_cost"][0]
    total_time       = metadata_df["total_time"][0]
    num_of_intervals = metadata_df["num_of_intervals"][0]

	#Export graphs
    num_of_params = 7
    data_path     = Path(__file__).parents[1] / "data" / "waveform_plots" / Path(*FOLDER_PATH.parts[-num_of_params:])
    data_path.mkdir(parents=True, exist_ok=True)

    theta_x_waveforms = pd.concat([pd.Series([theta_waveforms_df["theta_x"][0]]), theta_waveforms_df["theta_x"]])
    theta_y_waveforms = pd.concat([pd.Series(theta_waveforms_df["theta_y"][0]), theta_waveforms_df["theta_y"]])

    time_step         = total_time / num_of_intervals
    time_intervals    = np.linspace(0, total_time, num_of_intervals + 1)
    y_scale_limit     = 3.2

    plt.rc("font",**{"family":"serif","serif":["Palatino"]})
    plt.rc("text", usetex=True)

    #Graph theta_x_waveforms
    title_str = f"$\\theta_X$ Waveforms (Final Cost: {final_cost})"
    plt.figure(figsize=(15, 9), linewidth=2 * 1.15)
    plt.step(time_intervals, theta_x_waveforms, linewidth=2 * 1.15)
    plt.title(title_str, fontsize=20 * 1.15, pad=10)
    plt.xlabel("$\Omega t$", fontsize=20 * 1.15)
    plt.ylabel("$\\theta_X$", fontsize=20 * 1.15)
    plt.tick_params(labelsize=15 * 1.15)
    plt.ylim(-y_scale_limit, y_scale_limit)
    plt.savefig(data_path / "theta_x_waveforms_graph.svg")
    plt.savefig(data_path / "theta_x_waveforms_graph.png")

    #Graph theta_y_waveforms
    title_str = f"$\\theta_Y$ Waveforms (Final Cost: {final_cost})"
    plt.figure(figsize=(15, 9), linewidth=2 * 1.15)
    plt.step(time_intervals, theta_y_waveforms, linewidth=2 * 1.15)
    plt.title(title_str, fontsize=20 * 1.15, pad=10)
    plt.xlabel("$\Omega t$", fontsize=20 * 1.15)
    plt.ylabel("$\\theta_Y$", fontsize=20 * 1.15)
    plt.tick_params(labelsize=15 * 1.15)
    plt.ylim(-y_scale_limit, y_scale_limit)
    plt.savefig(data_path / "theta_y_waveforms_graph.svg")
    plt.savefig(data_path / "theta_y_waveforms_graph.png")

