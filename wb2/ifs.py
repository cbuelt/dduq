# File for extracting IFS data from weatherbench2 and saving it in a netCDF file.

import apache_beam   # Needs to be imported separately to avoid TypingError
import weatherbench2
import xarray as xr
import gcsfs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from netCDF4 import Dataset
import os

def filter_data(data_array, year, pred_horizon):
    date_range = pd.date_range(f"{year}-01-01", f"{year}-12-23", freq = "24h")
    data = data_array.sel(time = date_range).isel(prediction_timedelta = slice(0, pred_horizon), latitude=slice(None, None, -1))
    data = data.isel(latitude = namelist["lat_subset"], longitude = namelist["lon_subset"])
    return data

def get_filtered_data(year, pred_horizon = 32, ifs_path = "gs://weatherbench2/datasets/ifs_ens/2018-2022-1440x721_mean.zarr"):
    data_raw=xr.open_zarr(ifs_path)
    return filter_data(data_raw, year, pred_horizon)    

if __name__ == "__main__":
    namelist = dict()
    leadtimes = [t for t in range(0, 186 + 1,6)]
    namelist['lat_subset'] = np.arange(60,220)
    namelist['lon_subset'] = np.append(np.arange(1390, 1440), np.arange(0, 170))
    output_data_dir = "../mean/" 
    variables = ["10m_u_component_of_wind" , "10m_v_component_of_wind", "2m_temperature", "temperature", "geopotential"]
    years = [2018,2019,2020,2021,2022]

    for year in years:
        if year == 2020:
            n_ics = 358
        else:
            n_ics = 357

        # Load IFS data from weatherbench
        data_filtered = get_filtered_data(year)    
        
        # Create nc file for forecast
        filename_out = os.path.join(output_data_dir, f"{year}.nc" )
        # Check if file already exists
        if os.path.exists(filename_out):
            os.remove(filename_out)
        # Create file
        f = Dataset(filename_out, "a")
        
        # Create dimensions
        f.createDimension("pert", None)
        f.createDimension("var", None)
        f.createDimension("lead_time", len(leadtimes))
        f.createDimension("ics", n_ics)
        f.createDimension("lat", len(namelist["lat_subset"]))
        f.createDimension("lon", len(namelist["lon_subset"]))
        
        # Create variables
        time = f.createVariable("lead_time", "i4", ("lead_time",))
        forecast = f.createVariable(
            "forecast", "f4", ("ics", "pert", "var", "lead_time", "lat", "lon")
        )
        time[:] = leadtimes
    
        for i, var in enumerate(variables):
            print(f"Starting to extract variable {var}")
            if var == "temperature":
                forecast[:,0,i] = data_filtered[var].sel(level = 850)
            elif var == "geopotential":
                forecast[:,0,i] = data_filtered[var].sel(level = 500)     
            else:
                forecast[:,0,i] = data_filtered[var]
            print(f"Finished variable {var}")
        
        f.close()