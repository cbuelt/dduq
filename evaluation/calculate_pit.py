# Description: Calculate PIT for all methods and save to file.
# Author: Christopher Bülte

import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import h5py
from statsmodels.distributions.empirical_distribution import ECDF
from isodisreg import idr
from itertools import product
from datetime import datetime
import scipy
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils.utils import (
    load_prediction_single,
    load_truth,
    get_normalization,
    get_shapes,
    load_ifs_forecast,
    load_prediction,
    load_drn_prediction,
)


def get_ensembles(
    n_ics: int,
    n_points: int,
    lead_times: int,
    var: int,
    lat_subset: np.ndarray,
    lon_subset: np.ndarray,
    ground_truth: xr.DataArray,
    output_file,
    year=2022,
) -> None:
    """Calculates the PIT for the ensemble predictions.

    Args:
        n_ics (int): Number of initial conditions.
        n_points (int): Number of gridpoints to evaluate.
        lead_times (int): Number of lead times.
        var (int): Variable index.
        lat_subset (np.ndarray): Latitude of chosen points.
        lon_subset (np.ndarray): Longitude of chosen points.
        ground_truth (xr.DataArray): Ground truth data.
        output_file: File to save results to.
        year (int, optional): Forecast year. Defaults to 2022.
    """
    # Zip ics and subset of gridpoints together
    ics_lat_comb = product(np.arange(0, n_ics), np.arange(0, n_points))
    # Get normalization constants
    mean, std = get_normalization()

    for ics, l in ics_lat_comb:
        for t, lead_time in enumerate(lead_times):
            # Load truth
            truth = ground_truth.isel(
                ics=ics,
                lead_time=lead_time,
                var=var,
                lat=lat_subset[l],
                lon=lon_subset[l],
            )
            # Renormalize truth
            truth = truth * std[var] + mean[var]

            # ECMWF IFS
            ifs = load_ifs_forecast(ics).isel(
                lead_time=lead_time,
                var=var,
                lat=lat_subset[l],
                lon=lon_subset[l],
            )
            output_file[ics, l, t, 0] = ECDF(ifs)(truth)

            # RNP
            rnp = load_prediction_single(year, ics, name="rnp").isel(
                pert=slice(1, 51),
                lead_time=lead_time,
                var=var,
                lat=lat_subset[l],
                lon=lon_subset[l],
            )
            output_file[ics, l, t, 1] = ECDF(rnp)(truth)

            # IFSP
            ifsp = load_prediction_single(year, ics, name="ifsp").isel(
                pert=slice(0, 50),
                lead_time=lead_time,
                var=var,
                lat=lat_subset[l],
                lon=lon_subset[l],
            )
            output_file[ics, l, t, 2] = ECDF(ifsp)(truth)

            # RFP
            rfp = load_prediction_single(year, ics, name="rfp").isel(
                pert=slice(1, 51),
                lead_time=lead_time,
                var=var,
                lat=lat_subset[l],
                lon=lon_subset[l],
            )
            output_file[ics, l, t, 3] = ECDF(rfp)(truth)


def load_training_data(
    years: list, step: int, var: int, normalize: bool = False
) -> xr.DataArray:
    """Loads the training data for the EasyUQ method for a list of years, lead time and variable.
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
            load_prediction(year, ensemble=False).isel(var=var, lead_time=step)
            for year in years
        ],
        dim="ics",
    )
    if normalize:
        mean, std = get_normalization()
        training_data = training_data * std[var] + mean[var]
    return training_data


def load_training_truth(
    years: list, step: int, var: int, normalize: bool = False
) -> xr.DataArray:
    """Loads the training target for the EasyUQ method for a list of years, lead time and variable.

    Args:
        years (list): List of training years.
        step (int): Forecast lead time.
        var (int): Variable index.
        normalize (bool, optional): Whether data needs to be renormalized. Defaults to False.

    Returns:
        xr.DataArray: Training target.
    """
    training_truth = xr.concat(
        [load_truth(year).isel(var=var, lead_time=step) for year in years], dim="ics"
    )
    if normalize:
        mean, std = get_normalization()
        training_truth = training_truth * std[var] + mean[var]
    return training_truth


def get_easyuq(
    n_ics: int,
    n_points: int,
    lead_times: int,
    var: int,
    lat_subset: np.ndarray,
    lon_subset: np.ndarray,
    ground_truth: xr.DataArray,
    output_file
) -> None:
    """ Calculates the PIT for the EasyUQ forecast.

    Args:
        n_ics (int): Number of initial conditions.
        n_points (int): Number of gridpoints to evaluate.
        lead_times (int): Number of lead times.
        var (int): Variable index.
        lat_subset (np.ndarray): Latitude of chosen points.
        lon_subset (np.ndarray): Longitude of chosen points.
        ground_truth (xr.DataArray): Ground truth data.
        output_file: File to save results to.
        year (int, optional): Forecast year. Defaults to 2022.
    """
    # Get normalization constants
    mean, std = get_normalization()
    # Parameters
    training_years = [2018, 2019, 2020, 2021]
    # Load evaluation data
    evaluation_data = load_prediction(2022, ensemble=False, name="rfp")

    # Iterate through gridpoints
    for l in range(n_points):
        for t, lead_time in enumerate(lead_times):
            # Create train and test data
            x_train = load_training_data(
                training_years, lead_time, var, normalize=False
            ).isel(lat=lat_subset[l], lon=lon_subset[l])
            y_train = load_training_truth(
                training_years, lead_time, var, normalize=True
            ).isel(lat=lat_subset[l], lon=lon_subset[l])

            # Evaluation data
            y_val = (
                ground_truth.isel(
                    lead_time=lead_time, var=var, lat=lat_subset[l], lon=lon_subset[l]
                )
                * std[var]
                + mean[var]
            )
            x_val = evaluation_data.isel(
                lead_time=lead_time, var=var, lat=lat_subset[l], lon=lon_subset[l]
            )

            fitted_idr = idr(
                y_train, pd.DataFrame({"fore": x_train.to_numpy()}, columns=["fore"])
            )
            preds_test = fitted_idr.predict(
                pd.DataFrame({"fore": x_val.to_numpy()}, columns=["fore"])
            )
            output_file[:, l, t, 4] = preds_test.pit(y_val.to_numpy())[0:n_ics]


def get_drn(
    n_ics: int,
    n_points: int,
    lead_times: int,
    var: int,
    lat_subset: np.ndarray,
    lon_subset: np.ndarray,
    ground_truth: xr.DataArray,
    output_file
) -> None:
    """ Calculates the PIT for the DRN forecast.

    Args:
        n_ics (int): Number of initial conditions.
        n_points (int): Number of gridpoints to evaluate.
        lead_times (int): Number of lead times.
        var (int): Variable index.
        lat_subset (np.ndarray): Latitude of chosen points.
        lon_subset (np.ndarray): Longitude of chosen points.
        ground_truth (xr.DataArray): Ground truth data.
        output_file: File to save results to.
        year (int, optional): Forecast year. Defaults to 2022.
    """
    # Get normalization constants
    mean, std = get_normalization()

    for t, lead_time in enumerate(lead_times):
        drn = load_drn_prediction(var, lead_time)
        for l in range(n_points):
            truth = ground_truth.isel(
                lead_time=lead_time,
                var=var,
                lat=lat_subset[l],
                lon=lon_subset[l],
            )
            # Renormalize truth
            truth = truth * std[var] + mean[var]
            # Load drn
            drn_pred = drn[:, lat_subset[l], lon_subset[l]]
            # Extract mu and sigma
            drn_pred[:, 1] = np.abs(drn_pred[:, 1])
            output_file[:, l, t, 5] = scipy.stats.norm(
                loc=drn_pred[:, 0], scale=drn_pred[:, 1]
            ).cdf(truth)[0:n_ics]


if __name__ == "__main__":
    # Define parameters and paths
    year = 2022
    idx = {"u10": 0, "v10": 1, "t2m": 2, "t850": 3, "z500": 4}
    n_ics, length, n_var, lat_range, lon_range = get_shapes(year)
    n_points = 10
    output_path = ".."
    filename = "pit.h5"
    # Names of methods
    methods = ["ECMWF IFS", "RNP", "IFSP", "RFP", "EasyUQ", "DRN"]
    # Choose specific lead times
    lead_times = [4, 12, 28]

    # Load truth and normalization
    ground_truth = load_truth(year)

    # Create subset of n gridpoints for evaluation
    lat_subset = np.random.randint(0, lat_range, size=n_points)
    lon_subset = np.random.randint(0, lon_range, size=n_points)

    # Create file
    file = h5py.File(output_path + filename, "a")

    for var in range(n_var):
        # Measure time
        start_time = datetime.now()
        
        # Create dataset for variable
        var_name = list(idx.keys())[var]
        try:
            pit = file.create_dataset(
                name=var_name,
                shape=(n_ics, n_points, len(lead_times), len(methods)),
                dtype=np.float32,
            )
        except:
            del file[var_name]
            pit = file.create_dataset(
                name=var_name,
                shape=(n_ics, n_points, len(lead_times), len(methods)),
                dtype=np.float32,
            )

        # Get pit transforms for each method
        get_ensembles(
            n_ics=n_ics,
            n_points=n_points,
            lead_times=lead_times,
            var=var,
            lat_subset=lat_subset,
            lon_subset=lon_subset,
            ground_truth=ground_truth,
            output_file=pit,
        )
        get_easyuq(
            n_ics=n_ics,
            n_points=n_points,
            lead_times=lead_times,
            var=var,
            lat_subset=lat_subset,
            lon_subset=lon_subset,
            ground_truth=ground_truth,
            output_file=pit,
        )
        get_drn(
            n_ics=n_ics,
            n_points=n_points,
            lead_times=lead_times,
            var=var,
            lat_subset=lat_subset,
            lon_subset=lon_subset,
            ground_truth=ground_truth,
            output_file=pit,
        )

        # Measure time
        end_time = datetime.now()
        print("Duration: for variable {}: {}".format(var_name, end_time - start_time))
    # Close file
    file.close()
