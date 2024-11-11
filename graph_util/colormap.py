#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LogNorm
from fractions import Fraction
from functools import reduce

#Conversion to column name to label
variable_label_dict = {
    "num_of_intervals": "Number of Time Intervals",
    "total_time": "Total Time",
    "drift_parameter": "Drift Parameter",
    "taylor_truncate_len": "Taylor Truncated Length"
}

def reduce_grape_data(grape_data_df):
    min_init_seed_cost_group   = ["num_of_intervals", "total_time", "drift_parameter", "taylor_truncate_len",  "target_state_seed"]
    min_init_seed_cost_data_df = grape_data_df.groupby(min_init_seed_cost_group, as_index=False)["final_cost"].min() #Cost is negative!

    avg_cost_group   = ["num_of_intervals", "total_time", "drift_parameter", "taylor_truncate_len"]
    avg_cost_data_df = min_init_seed_cost_data_df.groupby(avg_cost_group, as_index=False)["final_cost"].mean()

    avg_cost_data_df["final_cost"] = 1 - (-1 * avg_cost_data_df["final_cost"])
    reduced_grape_data_df          = avg_cost_data_df.rename(columns={"final_cost": "avg_infidelity"})

    return reduced_grape_data_df

def generate_colormap_plot(avg_infidelity_list, x_data, y_data, x_label, y_label, hilbert_dimension, remaining_variable_constants):
    #TODO: Make not global?
    plt.rc("font",**{"family":"serif","serif":["Palatino"]})
    plt.rc("text", usetex=True)

    #Create colormap graph
    spin = Fraction((hilbert_dimension - 1) / 2)

    colormap_figure, colormap_axis = plt.subplots(figsize=(15, 10), linewidth=2 * 1.15)
    
    colormap_graph = plt.pcolor(x_data, y_data, avg_infidelity_list, norm=LogNorm(), cmap=cm.jet)
    color_bar      = plt.colorbar(colormap_graph)

    title = f"Infidelity vs. {x_label} \& {y_label}" "\n" f"(Spin: {spin}"
    for remaining_variable, constant in remaining_variable_constants.items():
        title += f", {variable_label_dict[remaining_variable]}: {constant}"
    title += ")"

    colormap_axis.set_title(title, fontsize=20 * 1.15, pad=10, loc="center")
    color_bar.ax.tick_params(labelsize=15 * 1.15)
    colormap_axis.set_xlabel("Total Time", fontsize=20 * 1.15)
    colormap_axis.set_ylabel("Number of Time Intervals", fontsize=20 * 1.15)
    colormap_axis.set_xticks(x_data)
    colormap_axis.set_yticks(y_data)
    colormap_axis.tick_params(labelsize=15 * 1.15)

    return colormap_figure

def main():
    #Parse system arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_path", type=lambda p: Path(p).absolute())
    parser.add_argument("--x_name", choices=variable_label_dict, required=True)
    parser.add_argument("--y_name", choices=variable_label_dict, required=True)
    parser.add_argument("--intervals", nargs="?", type=int)
    parser.add_argument("--total_time", nargs="?", type=float)
    parser.add_argument("--drift_param", nargs="?", type=float)
    parser.add_argument("--taylor_truncate_len", nargs="?", type=int)
    args = parser.parse_args()	

    #Initialize constants
    FOLDER_PATH         = args.folder_path
    X_NAME              = args.x_name
    Y_NAME              = args.y_name
    NUM_OF_INTERVALS    = args.intervals
    TOTAL_TIME          = args.total_time
    DRIFT_PARAMETER     = args.drift_param
    TAYLOR_TRUNCATE_LEN = args.taylor_truncate_len

    #Validate argument inputs
    if(X_NAME == Y_NAME):
        raise Exception("X data and Y data cannot be the same")

    remaining_variables          = set(variable_label_dict) - {X_NAME, Y_NAME}
    remaining_variable_constants = dict() #Mapping from variable name string to its value to be held constant

    if("num_of_intervals" in remaining_variables):
        if(NUM_OF_INTERVALS == None):
            raise Exception("You must set num_of_intervals")

        remaining_variable_constants["num_of_intervals"] = NUM_OF_INTERVALS

    if("total_time" in remaining_variables):
        if(TOTAL_TIME == None):
            raise Exception("You must set total_time")
        remaining_variable_constants["total_time"] = TOTAL_TIME

    if("drift_parameter" in remaining_variables):
        if(DRIFT_PARAMETER == None):
            raise Exception("You must set drift_param")
        remaining_variable_constants["drift_parameter"] = DRIFT_PARAMETER

    if("taylor_truncate_len" in remaining_variables):
        if(TAYLOR_TRUNCATE_LEN == None):
            raise Exception("You must set taylor_truncate_len")
        remaining_variable_constants["taylor_truncate_len"] = TAYLOR_TRUNCATE_LEN

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

    #Read metadata to get Hilbert dimension
    try:
        metadata_df = pd.read_csv(FOLDER_PATH / "metadata.csv")
    except FileNotFoundError as exception:
        raise Exception("Cannot find metadata.csv in given path") from exception
    hilbert_dimension = metadata_df["hilbert_dim"][0]

    #Generate colormap plot
    #Only select the data for which the selected variables are held constant
    remaining_variables          = list(remaining_variables)
    remaining_variables_df_masks = [
        reduced_grape_data_df[remaining_variable] == remaining_variable_constants[remaining_variable]
        for remaining_variable in remaining_variables
    ]
    remaining_variables_total_df_mask = reduce(np.logical_or, remaining_variables_df_masks)

    sliced_reduced_grape_data_df = reduced_grape_data_df[remaining_variables_total_df_mask]

    x_data              = sliced_reduced_grape_data_df[X_NAME].unique()
    y_data              = sliced_reduced_grape_data_df[Y_NAME].unique()
    avg_infidelity_list = pd.pivot_table(sliced_reduced_grape_data_df, values="avg_infidelity", index=Y_NAME, columns=X_NAME).to_numpy().reshape(len(y_data), len(x_data))

    #Set x and y labels
    x_label = variable_label_dict[X_NAME]
    y_label = variable_label_dict[Y_NAME]

    colormap_plot = generate_colormap_plot(avg_infidelity_list, x_data, y_data, x_label, y_label, hilbert_dimension, remaining_variable_constants)

    colormap_file_prefix = f"x_{X_NAME}_y_{Y_NAME}"
    for remaining_variable, constant in remaining_variable_constants.items():
        colormap_file_prefix += f"_{remaining_variable}_{constant}"

    colormap_plot.savefig(FOLDER_PATH / f"{colormap_file_prefix}_colormap_plot.png")
    colormap_plot.savefig(FOLDER_PATH / f"{colormap_file_prefix}_colormap_plot.svg")

if __name__ == "__main__":
    main()

