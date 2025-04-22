from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def read_batch_res_table(
        fpath_in: str | Path
        ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    # Returns: ou_mean, ou_std, {pop: {metric: values}}

    # Read the CSV file into a pandas DataFrame
    fpath_in = Path(fpath_in)
    df = pd.read_csv(fpath_in)

    # Create a list of (ou_mean, ou_std) tuples
    ou_mean = df['ou_mean'].to_numpy()
    ou_std = df['ou_std'].to_numpy()

    # Extract pop_names
    rate_columns = [col for col in df.columns if col.endswith('_r')]
    pop_names = [col[:-2] for col in rate_columns]

    # Read pop rates and Cv's from df
    data = {}
    for pop in pop_names:
        data[pop] = {
            'Rate': df[f'{pop}_r'].to_numpy(),
            'CV': df[f'{pop}_cv'].to_numpy()
        }
    
    return ou_mean, ou_std, data