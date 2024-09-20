# Basics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
import xarray as xr
import h5py

# Helpful
import time
import datetime
import itertools
from itertools import product

# My Methods
import importlib
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
from utils.utils import load_prediction, load_truth


# Load all data
data_path = "../mean/"
pred_2018 = load_prediction(year = 2018, ensemble = False, path = data_path)
pred_2019 = load_prediction(year = 2019, ensemble = False, path = data_path)
pred_2020 = load_prediction(year = 2020, ensemble = False, path = data_path)
pred_2021 = load_prediction(year = 2021, ensemble = False, path = data_path)

truth_2018 = load_truth(year = 2018)
truth_2019 = load_truth(year = 2019, ifs = True)
truth_2020 = load_truth(year = 2020)
truth_2021 = load_truth(year = 2021)


# Define Variable Names
var_names = ["u10", "v10", "t2m", "t850", "z500"]

# Calculate Mean and Std and create new file 
for var in range(len(var_names)):
    # Set up for file
    start_time = time.time()
    path = "../drn/data/train_test/" + var_names[var] + '_train.h5'
    f = h5py.File(path, "a")
    name_train = var_names[var] + '_train'
    name_truth = var_names[var] + '_truth'
    
    
    # Concatening the different years
    x_train = xr.concat([pred_2018.isel(var = var),
                             pred_2019.isel(var = var),
                             pred_2020.isel(var = var),
                             pred_2021.isel(var = var)],
                             dim = "ics")

    x_train = x_train.transpose(
    "ics", "lead_time", "lat", "lon"
    ).expand_dims(dim = {"mean_std":1}, axis = 4)

    y_train = xr.concat([truth_2018.isel(var = var),
                         truth_2019.isel(var = var),
                         truth_2020.isel(var = var),
                         truth_2021.isel(var = var)],
                         dim = "ics")
    
    n_days, n_lead_times, lat, long, mean_var  = x_train.shape
    
    half_time = time.time()
    time_difference_half = half_time - start_time
    hours = int(time_difference_half // 3600)
    minutes = int((time_difference_half % 3600) // 60)
    seconds = int(time_difference_half % 60)
    formatted_time_half = f" Round {var} finished concatenation in:{hours} hours, {minutes} minutes, {seconds} seconds"
    print(f"{formatted_time_half}")
    
    # Create those files
    try:
        train = f.create_dataset(
            name_train,
            data = x_train,
            shape=(n_days, n_lead_times, lat, long, mean_var),
            dtype=np.float32,
        )
    except:
        del f[name_train]
        train = f.create_dataset(
            name_train,
            data = x_train,
            shape=(n_days, n_lead_times, lat, long, mean_var),
            dtype=np.float32,
        )
    try:
        truth = f.create_dataset(
            name_truth,
            data = y_train,
            shape=(n_days, n_lead_times, lat, long),
            dtype=np.float32,
        )
    except:
        del f[name_truth]
        truth = f.create_dataset(
            name_truth,
            data = y_train,
            shape=(n_days, n_lead_times, lat, long),
            dtype=np.float32,
        )

        
    end_time = time.time()
    time_difference = end_time - start_time
    hours = int(time_difference // 3600)
    minutes = int((time_difference % 3600) // 60)
    seconds = int(time_difference % 60)
    f.close()
    formatted_time = f" Round {var} finished in:{hours} hours, {minutes} minutes, {seconds} seconds"
    print(f"{formatted_time}")
    













