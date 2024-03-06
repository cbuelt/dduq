# Description: Calculates the CRPS for the ensemble prediction of a model forecast.
# Author: Christopher Bülte

import h5py
import sys
import os
import numpy as np
from datetime import datetime
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils.utils import load_prediction_single, load_truth, get_normalization, get_shapes, get_crps_per_variable

if __name__ == '__main__':    
    # Load truth and prediction
    year = 2022
    ground_truth = load_truth(year)
    mean, std = get_normalization()

    # Set outputpath and filename
    output_path = ".."
    filename = "pangu_ifsp_crps.h5"
    n_ics, length, n_var, lat_range, lon_range = get_shapes(year)
    # Create output file
    file = h5py.File(output_path + filename, 'a')
    try:
        crps = file["crps"]    
    except:
        crps = file.create_dataset(name = "crps", shape = (n_ics, length, n_var, lat_range, lon_range), dtype = np.float32)  

    # Iterate through variables and variables
    for ics in range(n_ics):
        # Measure time
        start_time = datetime.now() 
        for var in range(n_var):    
            truth = ground_truth.isel(ics = ics, var = var)
            pred = load_prediction_single(year = year, ics = ics, name = "ifsp").isel(var = var, pert = slice(0,50))
            
            # Calculate CRPS
            crps_per_var = get_crps_per_variable(truth, pred, var)
            # Save
            crps[ics,:,var] = crps_per_var

        time_elapsed = datetime.now() - start_time 
        print(f'Time elapsed  for ICS {ics}: (hh:mm:ss.ms) {time_elapsed}')
            

    # Close file
    file.close()
    