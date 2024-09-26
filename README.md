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
Run Simple GRAPE algorithm
```
$ ./simple_grape/main.py -h
usage: main.py [-h] [--hilbert_dim [HILBERT_DIM]] [--intervals [INTERVALS]] [--total_time [TOTAL_TIME]] [--drift_param [DRIFT_PARAM]] [--init_seed [INIT_SEED]]
               [--target_state_seed [TARGET_STATE_SEED]]

options:
  -h, --help            show this help message and exit
  --hilbert_dim [HILBERT_DIM]
  --intervals [INTERVALS]
  --total_time [TOTAL_TIME]
  --drift_param [DRIFT_PARAM]
  --init_seed [INIT_SEED]
  --target_state_seed [TARGET_STATE_SEED]
```

## Graph Util
Generate waveform graphs
```
$ ./graph_util/main.py -h
usage: main.py [-h] folder_path

positional arguments:
  folder_path

options:
  -h, --help   show this help message and exit
```
