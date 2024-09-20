# Purpose: Run EasyUQ on the data-driven model output.
# The method is trained and evaluated per grid point, variable, lead time.
# Authors: Christopher Bülte

import numpy as np
import pandas as pd
import xarray as xr
from isodisreg import idr
import multiprocessing as mp
from itertools import product
import time
import h5py
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils.utils import load_prediction, load_truth, get_normalization, get_shapes


def load_training_data(
    years: list, step: int, var: int, normalize: bool = False, path: str = ""
) -> xr.DataArray:
    """Loads the training data for a list of years, lead time and variable.
    Args:
        years (list): List of training years.
        step (int): Forecast lead time.
        var (int): Variable index.
        normalize (bool, optional): Whether data needs to be renormalized. Defaults to False.
    Returns:
        xr.DataArray: Training data.
    """
    training_data = xr.concat(
        [
            load_prediction(year, ensemble=False, path = path).isel(var=var, lead_time=step)
            for year in years
        ],
        dim="ics",
    )
    if normalize:
        mean, std = get_normalization()
        training_data = training_data * std[var] + mean[var]
    return training_data


def load_training_truth(
    years: list, step: int, var: int, normalize: bool = False, ifs:bool = False,
) -> xr.DataArray:
    """Loads the training target for a list of years, lead time and variable.

    Args:
        years (list): List of training years.
        step (int): Forecast lead time.
        var (int): Variable index.
        normalize (bool, optional): Whether data needs to be renormalized. Defaults to False.

    Returns:
        xr.DataArray: Training target.
    """
    training_truth = xr.concat(
        [load_truth(year, ifs = ifs).isel(var=var, lead_time=step) for year in years], dim="ics"
    )
    if normalize:
        mean, std = get_normalization()
        training_truth = training_truth * std[var] + mean[var]
    return training_truth

    
def get_expected_value(prediction, q_steps = 0.0001):
    quantiles = np.arange(q_steps, 1, q_steps)
    quantile_pred = prediction.qpred(quantiles = quantiles)
    expected_value = (quantile_pred*q_steps).sum(axis = 1)
    return expected_value    


def get_standard_deviation(prediction, exp, q_steps = 0.0001):
    quantiles = np.arange(q_steps, 1, q_steps)
    quantile_pred = prediction.qpred(quantiles = quantiles)
    sd = np.sqrt(np.sum(np.power(quantile_pred - np.expand_dims(exp, axis = 1),2)*q_steps, axis = 1))
    return sd


def run_easy_uq(y_train:np.ndarray, x_train:np.ndarray, y_test:np.ndarray, x_test:np.ndarray, lat:int, lon:int)-> tuple:
    """ Trains the EasyUQ model and evaluates it on the test data, returning the expected value, standard deviation and CRPS.

    Args:
        y_train (np.ndarray): Training target.
        x_train (np.ndarray): Training data.
        y_test (np.ndarray): Evaluation target.
        x_test (np.ndarray): Evaluation data.
        lat (int): Latitude index.
        lon (int): Longitude index.

    Returns:
        tuple: Tuple containing the expected value, standard deviation, crps, latitude and longitude.
    """
    # Run idr
    fitted_idr = idr(y_train, pd.DataFrame(x_train.to_numpy()))
    preds_test = fitted_idr.predict(pd.DataFrame(x_test.to_numpy()))
    # Expected value
    exp = get_expected_value(preds_test, q_steps = 0.0001)
    # Standard deviation
    sd = get_standard_deviation(preds_test, exp, q_steps = 0.0001)
    # CRPS
    crps = np.array(preds_test.crps(y_test))
    return (exp, sd, crps, lat, lon)


def collect_result(result: tuple) -> None:
    """Collects the results from the multiprocessing pool and stores them in the results array.
    Args:
        result (tuple): Tuple containing the results, latitude and longitude.
    """
    global results
    exp, sd, crps, lat, lon = result
    results[0, :, lat, lon] = exp
    results[1, :, lat, lon] = sd
    results[2, :, lat, lon] = crps


if __name__ == "__main__":
    # Parameters
    ifs = True # For filtering wrong values in ifs forecast
    years = [2018, 2019, 2020, 2021]
    mean, std = get_normalization()
    n_ics, length, n_var, lat_range, lon_range = get_shapes(year = 2022)
    # Get grid of lat,lon
    lat_lon_comb = [
        element for element in product(np.arange(lat_range), np.arange(lon_range))
    ]

    # Define Number of processes to take place
    print(f"Number of cores: {len(os.sched_getaffinity(0))}")
    n_proc = len(os.sched_getaffinity(0))

    # Load truth and evaluation data
    data_path = "../mean/"
    truth = load_truth(year = 2022)
    evaluation_data = load_prediction(year = 2022, ensemble=False, path = data_path)

    # Create file
    output_path = "../results/ifs/"
    filename = "ifs_eq.h5"
    f = h5py.File(output_path + filename, "a")
    try:
        crps_eq = f["eq"]
    except:
        crps_eq = f.create_dataset(
            name="eq",
            shape=(3, n_ics, length, n_var, lat_range, lon_range),
            dtype=np.float32,
        )
        
    # Run loop across lead times and variables    
    for var in range(n_var):
        for step in range(1, length):        
            print(f"Running combination {var},{step}")
            # Create results array for asynch computing
            results = np.zeros(shape=(3, n_ics, lat_range, lon_range))
            t1 = time.time()

            # Create train and test data
            x_train = load_training_data(years, step, var, normalize=False, path = data_path)
            y_train = load_training_truth(years, step, var, normalize=True, ifs = True)

            # Validation data
            y_val = truth.isel(lead_time=step, var=var) * std[var] + mean[var]
            x_val = evaluation_data.isel(lead_time=step, var=var)

            # Run multiprocessing
            pool = mp.Pool(n_proc)
            for lat, lon in lat_lon_comb:
                pool.apply_async(
                    run_easy_uq,
                    args=(
                        y_train[:, lat, lon],
                        x_train[:, lat, lon],
                        y_val[:, lat, lon],
                        x_val[:, lat, lon],
                        lat,
                        lon,
                    ),
                    callback=collect_result,
                )
            pool.close()
            pool.join()
            t2 = time.time()
            crps_eq[:, :, step, var] = results
            print(f"Elapsed time for combination {var},{step}: {t2-t1}")
            print(np.mean(results[2]))
    f.close()
    print("File closed")
