#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd

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

    #Extract all data
    root_raw_data_path = Path(__file__).parents[1] / "data" / "grape_data" / f"{HILBERT_DIMENSION}_dim"

    num_of_intervals_list  = []
    total_time_list        = []
    target_state_seed_list = []
    init_seed_list         = []
    final_cost_list        = []
    for num_intervals_path in root_raw_data_path.glob("*"):
        for total_time_path in num_intervals_path.glob("*"):
            taylor_truncate_len_path = total_time_path / f"drift_param_{DRIFT_PARAMETER}" / f"taylor_len_{TAYLOR_TRUNCATE_LEN}"

            for target_state_seed_path in taylor_truncate_len_path.glob("*"):
                for init_seed_path in target_state_seed_path.glob("*"):
                    metadata_df = pd.read_csv(init_seed_path / "metadata.csv")

                    #Extract necessary metadata
                    num_of_intervals  = int(metadata_df["num_of_intervals"][0])
                    total_time        = float(metadata_df["total_time"][0])
                    target_state_seed = int(metadata_df["target_state_seed"][0])
                    init_seed         = int(metadata_df["init_seed"][0])
                    final_cost        = float(metadata_df["final_cost"][0])

                    num_of_intervals_list.append(num_of_intervals)
                    total_time_list.append(total_time)
                    target_state_seed_list.append(target_state_seed)
                    init_seed_list.append(init_seed)
                    final_cost_list.append(final_cost)

    #Export data
    data_path = Path(__file__).parents[1] / "data" / "contour_plots"
    data_path.mkdir(parents=True, exist_ok=True)

    cost_data_df = pd.DataFrame({
        "num_of_intervals"  : num_of_intervals_list,
        "total_time"        : total_time_list,
        "target_state_seed" : target_state_seed_list,
        "init_seed"         : init_seed_list,
        "final_cost"        : final_cost_list
    })
    cost_data_df.to_csv(data_path / f"{HILBERT_DIMENSION}_dim_drift_param_{DRIFT_PARAMETER}_taylor_len_{TAYLOR_TRUNCATE_LEN}_contour_data.csv", index=False)

