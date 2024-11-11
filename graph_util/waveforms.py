#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def generate_waveforms_plot(theta_waveforms, final_cost, total_time, num_of_intervals, y_scale_limit=3.2):
    time_intervals = np.linspace(0, total_time, num_of_intervals + 1)

    #TODO: Make not global?
    plt.rc("font",**{"family":"serif","serif":["Palatino"]})
    plt.rc("text", usetex=True)

    #Graph theta waveforms
    waveforms_figure, waveforms_axis = plt.subplots(figsize=(15, 9), linewidth=2 * 1.15)

    waveforms_axis.set_title(f"$\\theta$ Waveforms (Final Cost: {final_cost})", fontsize=20 * 1.15, pad=10)
    waveforms_axis.step(time_intervals, theta_waveforms, linewidth=2 * 1.15)
    waveforms_axis.set_xlabel("$\Omega t$", fontsize=20 * 1.15)
    waveforms_axis.set_ylabel("$\\theta$", fontsize=20 * 1.15)
    waveforms_axis.tick_params(labelsize=15 * 1.15)
    waveforms_axis.set_ylim(-y_scale_limit, y_scale_limit)

    return waveforms_figure

def main():
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=lambda p: Path(p).absolute())
    args = parser.parse_args()	

    #Initialize constants
    FOLDER_PATH = args.folder_path

    #Read raw data
    try:
        metadata_df = pd.read_csv(FOLDER_PATH / "metadata.csv")
    except FileNotFoundError as exception:
        raise Exception("Cannot read metadata.csv in given path") from exception
    
    try:
        theta_waveforms_df = pd.read_csv(FOLDER_PATH / "theta_waveforms.csv")
    except FileNotFoundError as exception:
        raise Exception("Cannot read theta_waveforms.csv (Only single_run.py produces waveform data)") from exception

    #Extract necessary metadata
    final_cost       = metadata_df["final_cost"][0]
    total_time       = metadata_df["total_time"][0]
    num_of_intervals = metadata_df["num_of_intervals"][0]

    #Extract theta waveforms data
    theta_waveforms = pd.concat([pd.Series([theta_waveforms_df["theta"][0]]), theta_waveforms_df["theta"]]) #Prepend first initial data point, so first interval is shown

    #Generate waveforms plot
    waveforms_plot = generate_waveforms_plot(theta_waveforms, final_cost, total_time, num_of_intervals)

    #Export plots
    waveforms_plot.savefig(FOLDER_PATH / "waveforms.png")
    waveforms_plot.savefig(FOLDER_PATH / "waveforms.svg")

if __name__ == "__main__":
    main()
