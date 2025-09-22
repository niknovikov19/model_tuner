- Input: OU current (tau=10)
- Goal: map "unconnected" to "connected" firing rates (UC)
- "Unconnected" rates are obtained from isolated populations in a surrogate surrounding.
- Surrogate state: `state_1` (see `target_state_1.csv`).
- Mapping from (ou_mean, ou_std) inputs to "unconnected" rates is given by a 2-d grid (`r_cv_grid_20x20.nc`)
- Subconn is turned off.
- For exc. cells, gKDR is increased to avoid cell-level multistability (see `mech_changes_1.json`)
    It distorts f-I curves compared to experimental data, re-tuning is needed in future.
