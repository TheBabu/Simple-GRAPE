# Simple GRAPE
Test simple GRAPE algorithm

## Installation
1. Create virtual environment
   
   ```
   virtualenv venv
   source venv/bin/activate
   ```
2. Install
   
   ```
   pip install -e .
   ```
   
3. Run scripts
   ```
   ./[path_to_script]/main.py
   ```

## Installation (with poetry)
1. Install
   
   ```
   poetry install
   ```
2. Run scripts
   
   ```
   poetry run ./[path_to_script]/main.py
   ```
   or alternatively
   ```
   poetry shell
   ./[path_to_script]/main.py
   ```

## Simple GRAPE
Run single instance of simple GRAPE algorithm
```
$ ./simple_grape/single_run.py --help
usage: single_run.py [-h] [--hilbert_dim [HILBERT_DIM]] [--intervals [INTERVALS]] [--total_time [TOTAL_TIME]] [--drift_param [DRIFT_PARAM]] [--taylor_truncate_len [TAYLOR_TRUNCATE_LEN]]
                     [--init_seed [INIT_SEED]] [--target_state_seed [TARGET_STATE_SEED]] [--no_plot_waveforms] [--no_plot_density] [--plot_density_time_interval [PLOT_DENSITY_TIME_INTERVAL]]
                     [--check_grad]

options:
  -h, --help            show this help message and exit
  --hilbert_dim [HILBERT_DIM]
  --intervals [INTERVALS]
  --total_time [TOTAL_TIME]
  --drift_param [DRIFT_PARAM]
  --taylor_truncate_len [TAYLOR_TRUNCATE_LEN]
  --init_seed [INIT_SEED]
  --target_state_seed [TARGET_STATE_SEED]
  --no_plot_waveforms
  --no_plot_density
  --plot_density_time_interval [PLOT_DENSITY_TIME_INTERVAL]
  --check_grad
```

Run multiple instances of simple GRAPE algorithm
```
$ ./simple_grape/multiple_run.py --help
usage: multiple_run.py [-h] [--hilbert_dim [HILBERT_DIM]] [--num_of_intervals_interval (START END STEP)] [--total_time_interval (START END STEP)] [--drift_param_interval (START END STEP)]
                       [--taylor_truncate_len_interval (START END STEP)] [--init_seed_interval (START END STEP)] [--target_state_seed_interval (START END STEP)]
                       (--check_grad | --folder_name FOLDER_NAME) [--no_reduce_grape_data]

options:
  -h, --help            show this help message and exit
  --hilbert_dim [HILBERT_DIM]
  --num_of_intervals_interval (START END STEP)
  --total_time_interval (START END STEP)
  --drift_param_interval (START END STEP)
  --taylor_truncate_len_interval (START END STEP)
  --init_seed_interval (START END STEP)
  --target_state_seed_interval (START END STEP)
  --check_grad
  --folder_name FOLDER_NAME
  --no_reduce_grape_data
```

## Full Grape
Run single instance of full GRAPE algorithm
```
$ ./full_grape/single_run.py --help
usage: single_run.py [-h] [--hilbert_dim [HILBERT_DIM]] [--targets [TARGETS]] [--intervals [INTERVALS]] [--total_time [TOTAL_TIME]] [--drift_param [DRIFT_PARAM]]
                     [--taylor_truncate_len [TAYLOR_TRUNCATE_LEN]] [--init_seed [INIT_SEED]] [--target_states_seed [TARGET_STATES_SEED]] [--no_plot_waveforms] [--no_plot_density]
                     [--plot_density_time_interval [PLOT_DENSITY_TIME_INTERVAL]] [--check_grad]

options:
  -h, --help            show this help message and exit
  --hilbert_dim [HILBERT_DIM]
  --targets [TARGETS]
  --intervals [INTERVALS]
  --total_time [TOTAL_TIME]
  --drift_param [DRIFT_PARAM]
  --taylor_truncate_len [TAYLOR_TRUNCATE_LEN]
  --init_seed [INIT_SEED]
  --target_states_seed [TARGET_STATES_SEED]
  --no_plot_waveforms
  --no_plot_density
  --plot_density_time_interval [PLOT_DENSITY_TIME_INTERVAL]
  --check_grad
```

Run multiple instances of full GRAPE algorithm
```
$ ./full_grape/multiple_run.py --help
usage: multiple_run.py [-h] [--hilbert_dim [HILBERT_DIM]] [--targets [TARGETS]] [--num_of_intervals_interval (START END STEP)] [--total_time_interval (START END STEP)]
                       [--drift_param_interval (START END STEP)] [--taylor_truncate_len_interval (START END STEP)] [--init_seed_interval (START END STEP)]
                       [--target_states_seed_interval (START END STEP)] (--check_grad | --folder_name FOLDER_NAME) [--no_reduce_grape_data]

options:
  -h, --help            show this help message and exit
  --hilbert_dim [HILBERT_DIM]
  --targets [TARGETS]
  --num_of_intervals_interval (START END STEP)
  --total_time_interval (START END STEP)
  --drift_param_interval (START END STEP)
  --taylor_truncate_len_interval (START END STEP)
  --init_seed_interval (START END STEP)
  --target_states_seed_interval (START END STEP)
  --check_grad
  --folder_name FOLDER_NAME
  --no_reduce_grape_data
```

## Graph Util
Generate waveforms plot
```
$ ./graph_util/waveforms.py --help
usage: waveforms.py [-h] folder_path

positional arguments:
  folder_path

options:
  -h, --help   show this help message and exit
```

Generate density animation
```
$ ./graph_util/density.py --help
usage: density.py [-h] [--time_interval [TIME_INTERVAL]] folder_path

positional arguments:
  folder_path

options:
  -h, --help            show this help message and exit
  --time_interval [TIME_INTERVAL]
```

Generate colormap plot
```
$ ./graph_util/colormap.py --help
usage: colormap.py [-h] --x_name {num_of_intervals,total_time,drift_parameter,taylor_truncate_len} --y_name
                   {num_of_intervals,total_time,drift_parameter,taylor_truncate_len} [--intervals [INTERVALS]] [--total_time [TOTAL_TIME]] [--drift_param [DRIFT_PARAM]]
                   [--taylor_truncate_len [TAYLOR_TRUNCATE_LEN]]
                   folder_path

positional arguments:
  folder_path

options:
  -h, --help            show this help message and exit
  --x_name {num_of_intervals,total_time,drift_parameter,taylor_truncate_len}
  --y_name {num_of_intervals,total_time,drift_parameter,taylor_truncate_len}
  --intervals [INTERVALS]
  --total_time [TOTAL_TIME]
  --drift_param [DRIFT_PARAM]
  --taylor_truncate_len [TAYLOR_TRUNCATE_LEN]
```
