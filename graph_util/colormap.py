#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from fractions import Fraction

def reduce_grape_data(grape_data_df):
    min_init_seed_cost_group   = ["num_of_intervals", "total_time", "drift_parameter", "taylor_truncate_len",  "target_state_seed"]
    min_init_seed_cost_data_df = grape_data_df.groupby(min_init_seed_cost_group, as_index=False)["final_cost"].min() #Cost is negative!

    avg_cost_group   = ["num_of_intervals", "total_time", "drift_parameter", "taylor_truncate_len"]
    avg_cost_data_df = min_init_seed_cost_data_df.groupby(avg_cost_group, as_index=False)["final_cost"].mean()

    avg_cost_data_df["final_cost"] = 1 - (-1 * avg_cost_data_df["final_cost"])
    reduced_grape_data_df          = avg_cost_data_df.rename(columns={"final_cost": "avg_infidelity"})

    return reduced_grape_data_df

# def generate_colormap_plot():
#     #TODO: Make not global?
#     plt.rc("font",**{"family":"serif","serif":["Palatino"]})
#     plt.rc("text", usetex=True)
#
#     #Create colormap graph
#     spin = Fraction((HILBERT_DIMENSION - 1) / 2)
#
#     plt.figure(figsize=(15, 10), linewidth=2 * 1.15)
#     plt.title(f"Infidelity vs. Total Time \& Number of Time Intervals (Spin: {spin})", fontsize=20 * 1.15, pad=10)
#     colormap_plt = plt.pcolor(total_time_list, num_of_intervals_list, avg_infidelity_list, norm=LogNorm(), cmap=cm.jet)
#     color_bar   = plt.colorbar(colormap_plt)
#     color_bar.ax.tick_params(labelsize=15 * 1.15)
#     plt.xlabel("Total Time", fontsize=20 * 1.15)
#     plt.ylabel("Number of Time Intervals", fontsize=20 * 1.15)
#     plt.xticks(total_time_list)
#     plt.yticks(num_of_intervals_list)
#     plt.tick_params(labelsize=15 * 1.15)
#     plt.savefig(FOLDER_PATH / f"{colormap_file_prefix}_graph.svg")
#     plt.savefig(FOLDER_PATH / f"{colormap_file_prefix}_graph.png")

def main():
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=lambda p: Path(p).absolute())
    args = parser.parse_args()	

    #Initialize constants
    FOLDER_PATH = args.folder_path

    #Try to read reduced GRAPE data, if not try to read unreduced GRAPE data and reduce it
    try:
        reduced_grape_data_df = pd.read_csv(FOLDER_PATH / "reduced_grape_data.csv")
    except FileNotFoundError:
        pass
    try:
        grape_data_df = pd.read_csv(FOLDER_PATH / "grape_data.csv")

        reduced_grape_data_df = reduce_grape_data(grape_data_df)

        #Export reduced_grape_data
        reduced_grape_data_df.to_csv(FOLDER_PATH / "reduced_grape_data.csv", index=False)
    except FileNotFoundError:
        raise Exception("Cannot find grape_data.csv")

if __name__ == "__main__":
    main()

