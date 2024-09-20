# Description: Calculate the bias for the different methods and save it in a file.
# Author: Christopher Bülte

import numpy as np
import h5py
from datetime import datetime
import sys
import os
import xarray as xr
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils.utils import load_prediction_single, load_truth, get_normalization, get_shapes, load_drn_prediction, load_ifs_forecast


if __name__ == "__main__":
    # Define parameters and paths
    year = 2022
    idx = {"u10": 0, "v10": 1, "t2m": 2, "t850": 3, "z500": 4}
    n_ics, length, n_var, lat_range, lon_range = get_shapes(year)
    output_path = "../results/ifs/"
    filename = "bias.h5"
    methods = ["ECMWF IFS", "RNP", "IFSP", "RFP", "EasyUQ", "DRN"]
    lead_times = [4, 12, 28]

    # Set paths
    drn_path = "../drn/results/preds/"

    # Load truth and normalization
    ground_truth = load_truth(year)
    # Get normalization constants
    mean, std = get_normalization()
    
    # Create file
    file = h5py.File(output_path + filename, "a")

   
    for var in range(n_var):    
        # Measure time
        start_time = datetime.now()
    
        # Create dataset for variable
        var_name = list(idx.keys())[var]
        try:
            bias = file.create_dataset(
            name=var_name,
            shape=(len(methods), len(lead_times), lat_range, lon_range),
            dtype=np.float32,
        )
        except:
            del file[var_name]
            bias = file.create_dataset(
                name=var_name,
                shape=(len(methods), len(lead_times), lat_range, lon_range),
                dtype=np.float32,
            )

        # Calculate bias for ensemble predictions
        for t, lead_time in enumerate(lead_times):
            for ic in range(n_ics):
                # Load Truth
                truth = ground_truth.isel(ics = ic, lead_time = lead_time, var = var)
                truth = truth * std[var] + mean[var]

                # ECMWF IFS
                ifs = load_ifs_forecast(ic).isel(
                        pert=slice(0, 50),
                        lead_time=lead_time,
                        var=var).mean(dim = ["pert"])
                bias[0, t] += (truth - ifs)/n_ics
                
        
                # RNP
                rnp = load_prediction_single(year, ic, name = "rnp").isel(
                        pert=slice(1, 51),
                        lead_time=lead_time,
                        var=var).mean(dim = ["pert"])
                bias[1, t] += (truth - rnp)/n_ics
        
                # IFSP
                ifsp = load_prediction_single(year, ic, name = "ifsp").isel(
                        pert=slice(0, 50),
                        lead_time=lead_time,
                        var=var).mean(dim = ["pert"])
                bias[2, t] += (truth - ifsp)/n_ics
        
        
                # RFP
                rfp = load_prediction_single(year, ic, name = "rfp").isel(
                        pert=slice(1, 51),
                        lead_time=lead_time,
                        var=var).mean(dim = ["pert"])
                bias[3, t] += (truth - rfp)/n_ics          


            # Calculate bias for EasyUQ
            truth = ground_truth.isel(lead_time = lead_time, var = var)
            truth = truth * std[var] + mean[var]
            eq = xr.open_dataset(output_path + "ifs_eq.h5").eq
            eq_mean = eq[0,:,lead_time,var]
            bias[4,t] = (truth - eq_mean.data).mean(dim = ["ics"])   
        
            
            # Calculate bias for DRN
            drn = load_drn_prediction(var, lead_time, path = drn_path)
            # Get mean
            drn_mean = drn[:,:,:,0]
            bias[5,t] = (truth - drn_mean.data).mean(dim = ["ics"])   


            
        # Close file
        end_time = datetime.now()
        print('Duration: for variable {}: {}'.format(var_name, end_time - start_time))        
    file.close()