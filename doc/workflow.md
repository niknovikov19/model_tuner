## Find working range for ou_mean (automatic)

### Config Description

#### SimManager side (local)

Repo: `model_tuner`
    Ref: `<MAIN:MODEL_TUNER>`
    Path ref: `<MAIN:MODEL_TUNER_PATH>`

Script: `proto\oumean_range_search\test_range_search_1.py`
    Ref: `<MAIN:OPT_SCRIPT>`

Local root folder: `dirpath_base_local`:
    Value: `<MAIN:MODEL_TUNER_PATH>\model_tuner\test_data\range_search`
    Ref: `<MAIN:BASE_DIR>`

Subfolder for a group of experiments: `exp_name_base`
    E.g.: `subnet_state1_mech1_nosub_wmult_0.1` (for models with 10% connectivity)
    Ref: `<MAIN:EXP_NAME_BASE>`

Sub-subfolder for an experiment: `exp_name`
    E.g.: `thal_tcalc_5_10`
    Ref: `<MAIN:EXP_NAME>`

Config files are located in:
    `<dirpath_base_local>\<MAIN:EXP_NAME_BASE>\<MAIN:EXP_NAME>`

**SSH config**: `ssh_params.yaml`

**Experiment config**: `exp_params.yaml`
    (ref: `<MAIN:EXP_PARAMS>`)
- `conda_env: netpyne_batch`
- `dirpath_hpc_base: <HPC:A1_OUINP_PATH>/exp_results/sim_manager_batch/exp_<MAIN:EXP_NAME_BASE>`
- `exp_name_base: sim_manager_batch/exp_<MAIN:EXP_NAME_BASE>`
- `exp_name: <MAIN:EXP_NAME>`
- `fpath_batch_script_hpc: <HPC:A1_OUINP_PATH>/opt_batch_script.py`

**Parameters of the range-detection algorithm**
    Located in `Task-specific part` secion of `<MAIN:OPT_SCRIPT>`.

**Populations to include**: `<MAIN:OPT_SCRIPT>.pop_names`
    (matches to `<MAIN:EXP_NAME>`)
    E.g.: `['TC', 'TCM', 'HTC', 'TI', 'TIM', 'IRE', 'IREM']`

Time window for rate calculation: `<MAIN:OPT_SCRIPT>.tcalc_win`

#### Simulation side (HPC)

Repo: `A1_OUinp`
    Path ref: `<HPC:A1_OUINP_PATH>`

Config files are located in:
    `<HPC:A1_OUINP_PATH>/exp_configs/sim_manager_batch/<MAIN:EXP_NAME_BASE>`

- `batch_params.py`
- `exp_cfg.py`
    - cfg.duration
    - cfg.wmult
    - cfg.addSubConn
    - cfg.mech_changes
    - cfg.ou_tau
- `mech_changes_1.json`
- `target_state_1.csv`


### Steps to create a new setup: range search for a model with 25% connectivity

#### SimManager side (local)

1. Create a folder for the group of experiments with 25% connectivity
    - The group name will be: `<MAIN:EXP_NAME_BASE> = exp_subnet_state1_mech1_nosub_wmult_0.25`
    - Create a folder: `<MAIN:BASE_DIR>\exp_subnet_state1_mech1_nosub_wmult_0.25`

2. Create a folder for an experiment
    - Each experiment runs the range search for a group of populations
    - You can do it for all pops in a single exp or split it into several pop groups
    - Our first experiment will contain the pops from layers 4 and 6
    - Experiment name: `<MAIN:EXP_NAME> = L46_tcalc_5_10`
    - Create a folder: `<MAIN:BASE_DIR>\exp_subnet_state1_mech1_nosub_wmult_0.25\L46_tcalc_5_10`

3. Copypaste `exp_params.yaml` and `ssh_params.yaml` from elsewhere to the newly created exp folder

4. Put the new `<MAIN:EXP_NAME_BASE>` and `<MAIN:EXP_NAME>` to `exp_params.yaml`:
    ```
    conda_env: netpyne_batch
    dirpath_hpc_base: <HPC:A1_OUINP_PATH>/exp_results/sim_manager_batch/exp_subnet_state1_mech1_nosub_wmult_0.25
    exp_name_base: sim_manager_batch/exp_subnet_state1_mech1_nosub_wmult_0.25
    exp_name: L46_tcalc_5_10
    fpath_batch_script_hpc: <HPC:A1_OUINP_PATH>/opt_batch_script.py
    ```

5. Modify params in `<MAIN:OPT_SCRIPT>`:
    - `exp_name_base = 'exp_subnet_state1_mech1_nosub_wmult_0.25'`
    - `exp_name = 'L46_tcalc_5_10'`
    - `pop_names = ['ITP4', 'ITS4', 'PV4', 'SOM4', 'VIP4', 'NGF4', 'IT6', 'CT6', 'PV6', 'SOM6', 'VIP6', 'NGF6']`

#### Simulation side (HPC)

6. Create a config folder for the new `<MAIN:EXP_NAME_BASE>`:
    `<HPC:A1_OUINP_PATH>/exp_configs/sim_manager_batch/exp_subnet_state1_mech1_nosub_wmult_0.25`

7. Copypaste files to the new folder from elsewhere (e.g. `exp_subnet_state1_mech1_nosub_wmult_0.1`):
    - `batch_params.py`
    - `exp_cfg.py`
    - `mech_changes_1.json`
    - `target_state_1.csv`

8. In `exp_cfg.py`, set:
    `cfg.wmult = 0.25`

#### SimManager side (local)

9. Run `<MAIN:OPT_SCRIPT>`


## Find working range for ou_mean (manual)

1. Create experiment group folder:
    `<HPC:A1_OUINP_PATH>/exp_configs/batch_i_ourange_unconn_state1_mech1_nosub_wmult_0.25`

2. Create experiment folder:
    `its4`

3. Copypaste config files into the exp folder

4. In `exp_cfg.py` set:
    `cfg.wmult = 0.25`

5. In `batch_params.py` set:
    ```
    BATCH_PARAMS = {
        'pop_name': 'ITS4',
        'ou_mean_range': (-0.1, 0.1),
        'ou_std': 0,
        'num_ou_points': 10,
        'tcalc_min': 5
        }
    ```
