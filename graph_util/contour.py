#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from fractions import Fraction

if __name__ == "__main__":
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hilbert_dim", nargs="?", default=2)
    parser.add_argument("--drift_param", nargs="?", default=1)
    parser.add_argument("--taylor_truncate_len", nargs="?", default=10)
    args = parser.parse_args()	

    #Initialize constants
    HILBERT_DIMENSION   = int(args.hilbert_dim)
    DRIFT_PARAMETER     = float(args.drift_param)
    TAYLOR_TRUNCATE_LEN = int(args.taylor_truncate_len)

    data_path = Path(__file__).parents[1] / "data" / "contour_plots"

    contour_file_prefix = f"{HILBERT_DIMENSION}_dim_drift_param_{DRIFT_PARAMETER}_taylor_len_{TAYLOR_TRUNCATE_LEN}_contour"
    cost_data_df        = pd.read_csv(data_path / f"{contour_file_prefix}_data.csv")

    min_init_seed_cost_data_df = cost_data_df.groupby(["num_of_intervals", "total_time", "target_state_seed"], as_index=False)["final_cost"].min() #Cost is negative!
    avg_cost_data_df           = min_init_seed_cost_data_df.groupby(["num_of_intervals", "total_time"], as_index=False)["final_cost"].mean()

    avg_cost_data_df["final_cost"] = 1 - (-1 * avg_cost_data_df["final_cost"])
    avg_infidelity_data_df         = avg_cost_data_df.rename(columns={"final_cost": "avg_infidelity"})

    num_of_intervals_list = avg_infidelity_data_df["num_of_intervals"].unique()
    total_time_list       = avg_infidelity_data_df["total_time"].unique()
    avg_infidelity_list   = np.array(avg_infidelity_data_df["avg_infidelity"]).reshape(len(total_time_list), len(num_of_intervals_list))

    #Export reduced contour data
    avg_infidelity_data_df.to_csv(data_path / f"{contour_file_prefix}_reduced_data.csv", index=False)

    #Export graph
    plt.rc("font",**{"family":"serif","serif":["Palatino"]})
    plt.rc("text", usetex=True)

    #Create contour graph
    spin = Fraction((HILBERT_DIMENSION - 1) / 2)

    plt.figure(figsize=(15, 10), linewidth=2 * 1.15)
    plt.title(f"Infidelity vs. Total Time \& Number of Time Intervals (Spin: {spin})", fontsize=20 * 1.15, pad=10)
    plt.contour(num_of_intervals_list, total_time_list, avg_infidelity_list, norm=LogNorm(), colors="k")
    contour_plt = plt.contourf(num_of_intervals_list, total_time_list, avg_infidelity_list, norm=LogNorm(), cmap=plt.cm.jet)
    color_bar   = plt.colorbar(contour_plt)
    color_bar.ax.tick_params(labelsize=15 * 1.15)
    plt.xlabel("Total Time", fontsize=20 * 1.15)
    plt.ylabel("Number of Time Intervals", fontsize=20 * 1.15)
    plt.yticks(num_of_intervals_list)
    plt.tick_params(labelsize=15 * 1.15)
    plt.savefig(data_path / f"{contour_file_prefix}_graph.svg")
    plt.savefig(data_path / f"{contour_file_prefix}_graph.png")

    #DEBUG
    # plt.show()

