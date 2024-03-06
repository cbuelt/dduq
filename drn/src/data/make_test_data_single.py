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
from utils import load_prediction, load_truth


# Load all data
pred_2022 = load_prediction(year = 2022, ensemble = False)
truth_2022 = load_truth(year = 2022)


# Define Variable Names
#dat_train_all = [dat_2018, dat_2019, dat_2020, dat_2021]
var_names = ["u10", "v10", "t2m", "t850", "z500"]

# Calc Mean and Std and create new test data file
for var in range(len(var_names)):
    # Set up for file
    start_time = time.time()
    path = "../drn/data/train_test/" + var_names[var] + '_test.h5'
    f = h5py.File(path, "a")
    name_test = var_names[var] + '_test'
    name_truth = var_names[var] + 'test_truth'
    
    x_test = pred_2022.isel(var=var)
    x_test = x_test.transpose(
    "ics", "lead_time", "lat", "lon"
    ).expand_dims(dim = {"mean_std":1}, axis = 4)

    y_test = truth_2022.isel(var = var)
    
    n_days, n_lead_times, lat, long, mean_var  = x_test.shape
    
    half_time = time.time()
    time_difference_half = half_time - start_time
    hours = int(time_difference_half // 3600)
    minutes = int((time_difference_half % 3600) // 60)
    seconds = int(time_difference_half % 60)
    formatted_time_half = f" Round {var} finished concatenation in:{hours} hours, {minutes} minutes, {seconds} seconds"
    print(f"{formatted_time_half}")
    
    # Create those files
    try:
        test = f.create_dataset(
            name_test,
            data = x_test,
            shape=(n_days, n_lead_times, lat, long, mean_var),
            dtype=np.float32,
        )
    except:
        del f[name_test]
        test = f.create_dataset(
            name_test,
            data = x_test,
            shape=(n_days, n_lead_times, lat, long, mean_var),
            dtype=np.float32,
        )
    try:
        truth = f.create_dataset(
            name_truth,
            data = y_test,
            shape=(n_days, n_lead_times, lat, long),
            dtype=np.float32,
        )
    except:
        del f[name_truth]
        truth = f.create_dataset(
            name_truth,
            data = y_test,
            shape=(n_days, n_lead_times, lat, long),
            dtype=np.float32,
        )
        
    end_time = time.time()
    time_difference = end_time - start_time
    hours = int(time_difference // 3600)
    minutes = int((time_difference % 3600) // 60)
    seconds = int(time_difference % 60)
    formatted_time = f" Round {var} finished in:{hours} hours, {minutes} minutes, {seconds} seconds"
    print(f"{formatted_time}")
    f.close()




