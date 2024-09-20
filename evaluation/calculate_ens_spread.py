# Description: Calculate ensemble spread for all methods and variables
# Author: Christopher Bülte

import numpy as np
from datetime import datetime
import sys
import os
import h5py
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils.utils import *


if __name__ == "__main__":
    # Define parameters and paths
    year = 2022
    idx = {"u10": 0, "v10": 1, "t2m": 2, "t850": 3, "z500": 4}
    n_ics, length, n_var, lat_range, lon_range = get_shapes(year)
    output_path = "../results/ifs/"
    filename = "ensemble_spread.h5"
    methods = ["ECMWF IFS", "RNP", "IFSP", "RFP", "EasyUQ", "DRN"]

    drn_path = "../drn/results/preds/"

    # Load truth and normalization
    mean, std = get_normalization()
    ground_truth = load_truth(year)

    # Create file
    file = h5py.File(output_path + filename, "a")

    # Create ensemble spread per variable
    for var in range(n_var):
        # Measure time
        start_time = datetime.now()
        
        # Create dataset for variable
        var_name = list(idx.keys())[var]
        try:
            ens_spread = file.create_dataset(
            name=var_name,
            shape=(n_ics, length, len(methods), 2),
            dtype=np.float32,
        )
        except:
            del file[var_name]
            ens_spread = file.create_dataset(
                name=var_name,
                shape=(n_ics, length, len(methods), 2),
                dtype=np.float32,
            )

        for ic in range(n_ics):
            # Truth value
            truth = ground_truth.isel(ics=ic, var=var)
            truth = truth * std[var] + mean[var]
            
            # ECMWF IFS
            ifs = load_ifs_forecast(ic).isel(var=var)
            
            rmse = np.sqrt(np.power(truth - ifs.mean(dim=["pert"]),2,).mean(dim=["lat", "lon"]))
            sd = ifs.std(dim=["pert"]).mean(dim=["lat", "lon"])
            
            ens_spread[ic,:,0,0] = rmse
            ens_spread[ic,:,0,1] = sd
            
            # RNP
            rnp = load_prediction_single(year, ic, name = "rnp").isel(pert=slice(1, 51),var=var)
            rmse = np.sqrt(np.power(truth - rnp.mean(dim=["pert"]), 2).mean(dim=["lat", "lon"]))
            sd = rnp.std(dim=["pert"]).mean(dim=["lat", "lon"])
            ens_spread[ic,:,1,0] = rmse
            ens_spread[ic,:,1,1] = sd
            
            # IFSP
            ifsp = load_prediction_single(year, ic, name = "ifsp").isel(pert=slice(1, 51),var=var)
            rmse = np.sqrt(np.power(truth - ifsp.mean(dim=["pert"]), 2).mean(dim=["lat", "lon"]))
            sd = ifsp.std(dim=["pert"]).mean(dim=["lat", "lon"])
            ens_spread[ic,:,2,0] = rmse
            ens_spread[ic,:,2,1] = sd        
            
            # RFP
            rfp = load_prediction_single(year, ic, name = "rfp").isel(pert=slice(1, 51),var=var)
            rmse = np.sqrt(np.power(truth - rfp.mean(dim=["pert"]), 2).mean(dim=["lat", "lon"]))
            sd = rfp.std(dim=["pert"]).mean(dim=["lat", "lon"])
            ens_spread[ic,:,3,0] = rmse
            ens_spread[ic,:,3,1] = sd

        for lead_time in range(length):
            # No values at zero for EasyUQ and DRN
            if lead_time>0:   
                # EasyUQ
                truth = ground_truth.isel(lead_time = lead_time, var = var)
                truth = truth * std[var] + mean[var]
                eq = xr.open_dataset(output_path + "ifs_eq.h5").eq
                eq_mean = eq[0,:,lead_time,var]
                rmse = np.sqrt(np.power(truth - eq_mean.data, 2).mean(dim=["lat", "lon"]))
                sd = eq[1,:,lead_time,var]
                ens_spread[:,lead_time,4,0] = rmse
                ens_spread[:,lead_time,4,1] = np.mean(sd, axis = (1,2))       
                
                # DRN
                drn = load_drn_prediction(var, lead_time, path = drn_path)
                # Get mean
                drn_mean = drn[:,:,:,0]
                rmse = np.sqrt(np.power(truth - drn_mean.data, 2).mean(dim=["lat", "lon"]))
                sd = np.abs(drn[:,:,:,1])
                ens_spread[:,lead_time,5,0] = rmse
                ens_spread[:,lead_time,5,1] = np.mean(sd, axis = (1,2))  

            else:
                ens_spread[:,lead_time,4,0] = 0
                ens_spread[:,lead_time,4,1] = 0 
                ens_spread[:,lead_time,5,0] = 0
                ens_spread[:,lead_time,5,1] = 0  
            
        end_time = datetime.now()
        print(f"Finished var {var} Duration: {end_time - start_time}")
    file.close()
