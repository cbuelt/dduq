# Helper functions for loading data and calculating CRPS.
# Author: Christopher Bülte

import numpy as np
import xarray as xr
import properscoring as ps
from datetime import datetime
import pandas as pd
import os


def get_normalization(
    path: str = "../stats/",
    vars: list = [0, 1, 2, 5, 14],
) -> tuple:
    """Loads normalizing constants for output variables from the specified path.

    Args:
        path (str, optional): Specifies directory where files are saved. Defaults to "".
        vars (list, optional): Index of specifies variables to be used.. Defaults to [0,1,2,5,14].

    Returns:
        tuple: Mean and standard deviation of variables.
    """
    mean = np.load(path + "global_means.npy").squeeze()[vars]
    std = np.load(path + "global_stds.npy").squeeze()[vars]
    return mean, std


def load_truth(
    year: int,
    path: str = "..",
) -> xr.DataArray:
    """Loads the ERA5 ground truth for a given year.

    Args:
        year (int): Specifies the year to be used.
        path (str, optional): Specifies the file path. Defaults to "".

    Returns:
        xr.DataArray: ERA5 ground truth with shape (ics, lead_times, variables, latitude, longitude)
    """
    if year == 2018:
        # Old file format
        file = f"{year}.nc"
        truth = (
            xr.open_dataset(path + file)
            .rename({"ic": "ics", "step": "lead_time"})
            .ground_truth
        )
    else:
        file = f"{year}.h5"
        truth = (
            xr.open_dataset(path + file)
            .rename(
                {
                    "phony_dim_0": "ics",
                    "phony_dim_2": "lead_time",
                    "phony_dim_3": "var",
                    "phony_dim_4": "lat",
                    "phony_dim_5": "lon",
                }
            )
            .truth
        )
    return truth


def get_shapes(year: int) -> tuple:
    """Get shapes of the data for a given year.

    Args:
        year (int): The respective year.

    Returns:
        tuple: Different data shapes.
    """
    shapes = load_truth(year).shape

    return shapes[0], shapes[1], shapes[2] - 1, shapes[3], shapes[4]


def load_ifs_forecast(
    ics: int,
    path: str = "..",
) -> xr.DataArray:
    """Loads the IFS forecast for a given initial condition.

    Args:
        ics (int): Initial condition corresponding to a datetime.
        path (str, optional): The file path. Defaults to "".

    Returns:
        xr.DataArray: IFS forecast in the shape (perturbations, lead_times, variables, latitude, longitude).
    """
    files = os.listdir(path)
    files.sort()
    ifs = (
        xr.open_dataset(path + files[ics])
        .rename(
            {"ics": "pert", "step": "lead_time",
                "latitude": "lat", "longitude": "lon"}
        )
        .fields.isel(pert=slice(0, 50))
    )
    return ifs


def load_prediction(
    year: int,
    path: str = "..",
    ensemble: bool = True,
) -> xr.DataArray:
    """Loads the model prediction for a given year.

    Args:
        year (int): Respective year.
        path (str, optional): File path. Defaults to "".
        ensemble (bool, optional): Whether to load the full ensemble or only the control forecast. Defaults to True.
        name (str, optional): Naming of the file. Defaults to "det".

    Returns:
        xr.DataArray: Model prediction in the shape (ics, perturbations, lead_times, variables, latitude, longitude).
    """
    file = f"{year}.nc"
    truth = xr.open_dataset(path + file).forecast
    if ensemble == False:
        truth = truth.isel(pert=0)
    return truth


def load_prediction_single(
    year: int,
    ics: int,
    path: str = "..",
    name: str = "rnp",
) -> xr.DataArray:
    """Loads a model prediction for a given year and specific initial condition.

    Args:
        year (int): Respective year
        ics (int):  Initial condition corresponding to a datetime.
        path (str, optional): File path. Defaults to "".
        name (str, optional): Name of the method/folder of the subfiles. Defaults to "rnp".

    Returns:
        xr.DataArray: Prediction in the shape (perturbations, lead_times, variables, latitude, longitude).
    """
    file_path = path + f"{name}/"
    datelist = pd.date_range(datetime(year, 1, 1), datetime(year, 12, 23)).strftime(
        "%Y-%m-%-d"
    )
    files = [file_path + "fc_" + date + ".nc" for date in datelist.tolist()]

    # Load prediction
    prediction = xr.open_dataset(files[ics]).forecast
    return prediction


def load_drn_prediction(
    var: int,
    lead_time: int,
    path: str = "..",
) -> np.ndarray:
    """Load the DRN prediction for a given variable and lead time.

    Args:
        var (int): Variable index.
        lead_time (int): Forecast lead time.
        path (str, optional): File path. Defaults to "".

    Returns:
        np.ndarray: Reshaped drn prediction.
    """
    n_ics, length, n_var, lat_range, lon_range = get_shapes(year=2022)
    idx = {"u10": 0, "v10": 1, "t2m": 2, "t850": 3, "z500": 4}
    file = f"DRN_{list(idx.keys())[var]}_lead_time_{(lead_time-1)}_preds.npy"
    drn = np.load(path + file).reshape(n_ics, lat_range, lon_range, 2)
    return drn


def get_crps_per_variable(
    truth: xr.DataArray,
    prediction: xr.DataArray,
    variable: int,
    normalize_truth: bool = True,
    normalize_pred: bool = False,
) -> np.ndarray:
    """Calculates the CRPS for a given variable across the data dimensions.

    Args:
        truth (xr.DataArray): Ground truth.
        prediction (xr.DataArray): Model forecast.
        variable (int): Variable index.
        normalize_truth (bool, optional): Whether to renormalize the ground truth. Defaults to True.
        normalize_pred (bool, optional): Whether to renormalize the model forecast. Defaults to False.

    Returns:
        np.ndarray: Array with all CRPS values.
    """
    # Normalize
    mean, std = get_normalization()
    if normalize_truth:
        truth = truth * std[variable] + mean[variable]
    if normalize_pred:
        prediction = prediction * std[variable] + mean[variable]

    # Transpose prediction and calculate CRPS
    prediction = prediction.transpose(..., "pert")
    crps = ps.crps_ensemble(truth, prediction)

    return crps
