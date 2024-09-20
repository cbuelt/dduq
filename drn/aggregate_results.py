# Purpose: Aggregate the output of the drn model and calculate the crps
# Authors: Christopher Bülte

import numpy as np
import pandas as pd
import xarray as xr
import properscoring as ps
import sys
import os
import h5py
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils.utils import load_drn_prediction, get_shapes

if __name__ == "__main__":
    # Get shapes
    n_ics, length, n_var, lat_range, lon_range = get_shapes(year = 2022)
    idx = {"u10": 0, "v10": 1, "t2m": 2, "t850": 3, "z500": 4}
    
    # Create file
    output_path = "../results/ifs/"
    drn_path = "../drn/results/preds/"
    filename = "ifs_drn.h5"
    f = h5py.File(output_path + filename, "a")
    try:
        drn_res = f["drn"]
    except:
        drn_res = f.create_dataset(
            name="drn",
            shape=(3, n_ics, length, n_var, lat_range, lon_range),
            dtype=np.float32,
        )

    # Aggregate results
    for var in range(n_var):
        for step in range(1,length):
            drn = load_drn_prediction(var = var, lead_time = step, path = drn_path)
            exp = drn[:,:,:,0]
            sd = drn[:,:,:,1]
            # Load score
            score_path = f"../drn/results/scores/DRN_{list(idx.keys())[var]}_lead_time_{(step-1)}_scores.npy"
            drn_score = np.load(score_path)

            # Write to file
            drn_res[0, :, step, var] = exp
            drn_res[1, :, step, var] = sd
            drn_res[2, :, step, var] = drn_score

    f.close()
            
                