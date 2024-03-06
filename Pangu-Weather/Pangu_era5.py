# Description: This script is used to generate the forecasts for the Pangu-Weather model.
# It can be used to generate the control forecast, random field perturbations, random noise perturbations and ECMWF perturbations.
# Forecasts are generated for a single day, specified by the date argument.
# Author: Christopher Bülte

import os
import numpy as np
import onnx
import onnxruntime as ort
import xarray as xr
import argparse
from constants import constants as cc
from Pangu_namelist import namelist
import calendar
import random
import datetime
from netCDF4 import Dataset
from perlin_numpy import generate_perlin_noise_2d


def randomdate(year: int, month: int) -> str:
    """Get a random date in a given month and year

    Args:
        year (int): _description_
        month (int): _description_

    Returns:
        str: Date in format yyyy-mm-dd.
    """
    dates = calendar.Calendar().itermonthdates(year, month)
    random_date = random.choice([date for date in dates if date.month == month])
    return datetime.datetime.strftime(random_date, "%Y-%m-%d")


def tke(ds_pl: xr.Dataset, ds_sfc: xr.Dataset) -> xr.DataArray:
    """Calculates the total kinetic energy of a dataset.

    Args:
        ds_pl (xr.Dataset): Dataset with pressure level data.
        ds_sfc (xr.Dataset): Dataset with surface data.

    Returns:
        xr.DataArray: Total kinetic energy.
    """
    weights = np.cos(np.deg2rad(ds_pl.lat))
    weights.name = "weights"

    Etot_1 = (
        (1.0 / cc["g"])
        * 0.5
        * (ds_pl.U**2 + ds_pl.V**2.0 + cc["Cp_d"] / cc["T_r"] * ds_pl.T**2.0)
    )
    Etot_1_w = Etot_1.weighted(weights)

    Etot_2 = (0.5 * cc["Rd"] * cc["T_r"] * cc["P_r"] / cc["g"]) * (
        np.log(ds_sfc.SP)
    ) ** 2.0
    Etot_2_w = Etot_2.weighted(weights)

    weighted_mean = Etot_1_w.mean(("lon", "lat")).integrate("plev") + Etot_2_w.mean(
        ("lon", "lat")
    )

    return weighted_mean


def get_stds() -> tuple:
    """Loads the standard deviations of the variables from the specified path.

    Returns:
        tuple: Standard deviations of variables.
    """
    path = namelist["std_path"]
    stds_pl = np.load(path + "pangu_stds_pl.npy")
    stds_sfc = np.load(path + "pangu_stds_sfc.npy")
    return stds_pl, stds_sfc


def perlin_noise_2d(
    res: int = 12,
    scales: list = [0.2, 0.1, 0.05],
    octaves: int = 3,
    persistence: float = 0.5,
) -> np.ndarray:
    """Generates 2d perlin noise for a single variable.

    Args:
        res (int, optional): _description_. Defaults to 12.
        scales (list, optional): _description_. Defaults to [0.2,0.1,0.05].
        octaves (int, optional): _description_. Defaults to 3.
        persistence (float, optional): _description_. Defaults to 0.5.

    Returns:
        np.ndarray: Generated perlin noise
    """
    shape = (720, 1440)
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _, i in enumerate(range(octaves)):
        noise += (
            scales[i]
            * amplitude
            * generate_perlin_noise_2d(shape, (frequency * res, frequency * res))
        )
        frequency *= 2
        amplitude *= persistence
    return noise


def get_perlin_noise_2d(n: int) -> np.ndarray:
    """Generate 2d Perlin noise for n variables.

    Args:
        n (int): Number of variables.

    Returns:
        np.ndarray: Generated perlin noise
    """
    result = np.zeros((n, 721, 1440))
    noise = np.array([perlin_noise_2d() for x in range(n)])
    result[:, 0:720, :] = noise
    return result


def inference(
    input_pl: np.ndarray, input_sfc: np.ndarray, index: int, file: Dataset
) -> None:
    """Runs the inference session for a given input. and saves the output to a specified file.

    Args:
        input_pl (np.ndarray): Pressure level data.
        input_sfc (np.ndarray): Surface data.
        index (int): Index of the perturbation.
        file (Dataset): The file to which the output is saved.
    """
    # Run the inference session
    input_pl_24, input_sfc_24 = input_pl, input_sfc
    for t, time in enumerate(leadtimes):
        if (t + 0) % 4 == 0 and t > 0:
            output_pl, output_sfc = ort_session_24.run(
                None, {"input": input_pl_24, "input_surface": input_sfc_24}
            )
            input_pl_24, input_sfc_24 = output_pl, output_sfc
        else:
            if t == 0:
                output_pl, output_sfc = input_pl, input_sfc
            else:
                output_pl, output_sfc = ort_session_6.run(
                    None, {"input": input_pl, "input_surface": input_sfc}
                )
        input_pl, input_sfc = output_pl, output_sfc

        for v, var_sfc in enumerate(namelist["sfc_out"]):
            file[index, v, t] = output_sfc[
                var_sfc, namelist["lat_subset"][:, None], namelist["lon_subset"]
            ]

        for var in list(namelist["pl_lev_out"].keys()):
            for pl in namelist["pl_lev_out"][var]:
                v += 1
                file[index, v, t] = output_pl[
                    var, pl, namelist["lat_subset"][:, None], namelist["lon_subset"]
                ]


def read_input(filename: str, date: str, engine: str = "netcdf4") -> tuple:
    """Reads the input data for a given date and filename.

    Args:
        filename (str): Specifies the filename.
        date (str): Specifies the date in format yyyy-mm-dd.
        engine (str, optional): Engine for reading files. Defaults to 'netcdf4'.

    Returns:
        tuple: Pressure level data, surface data, pressure levels, latitudes and longitudes.
    """
    # load pressure level data
    ds = (
        xr.open_dataset("%s/%s" % (input_data_dir, filename), engine=engine)
        .sel(time=date, plev=slice(None, None, -1))
        .astype(np.float32)
    )
    lat = ds["lat"].values
    lon = ds["lon"].values
    lev = ds["plev"].values
    input_pl = ds[namelist["pl_vars"]].to_array().to_numpy()

    # Load the surface data
    ds = (
        xr.open_dataset(
            "%s/%s" % (input_data_dir, filename.replace("pl", "sfc")), engine=engine
        )
        .sel(time=date)
        .astype(np.float32)
    )
    input_sfc = ds[namelist["sfc_vars"]].to_array().to_numpy()

    return input_pl, input_sfc, lev, lat, lon


def calculate_cf(
    input_pl: np.ndarray, input_sfc: np.ndarray, file_index: int, file: Dataset
) -> None:
    """Calculates the control forecast for a given input and saves the output to a specified file.

    Args:
        input_pl (np.ndarray): Pressure level data.
        input_sfc (np.ndarray): Surface data.
        index (int): Index of the perturbation.
        file (Dataset): The file to which the output is saved.
    """
    inference(input_pl, input_sfc, file_index, file)


def random_field_perturbation(input_pl:np.ndarray, input_sfc:np.ndarray, engine:str)-> tuple:
    """ Creates random field perturbations, based on two random dates.

    Args:
        input_pl (np.ndarray): Pressure level data.
        input_sfc (np.ndarray): Surface data.
        engine (str): Engine for reading files.

    Returns:
        tuple: Perturbed pressure level data and surface data.
    """
    # Find two random dates
    rand_yyyy1 = random.choice(namelist["ens_years"])
    rand_yyyy2 = random.choice(
        [yyyy for yyyy in namelist["ens_years"] if yyyy != rand_yyyy1]
    )
    date1 = randomdate(rand_yyyy1, int(date[5:7]))
    date2 = randomdate(rand_yyyy2, int(date[5:7]))
    print(f"Random dates for RFP: {date1}, {date2}")

    ds1 = (
        xr.open_dataset("%s/era5_%s_pl.nc" % (input_data_dir, date1[:7]), engine=engine)
        .sel(time=date1, plev=slice(None, None, -1))
        .astype(np.float32)
    )
    ds2 = (
        xr.open_dataset("%s/era5_%s_pl.nc" % (input_data_dir, date2[:7]), engine=engine)
        .sel(time=date2, plev=slice(None, None, -1))
        .astype(np.float32)
    )
    ds_pl = ds1 - ds2

    ds1 = (
        xr.open_dataset(
            "%s/era5_%s_sfc.nc" % (input_data_dir, date1[:7]), engine=engine
        )
        .sel(time=date1)
        .astype(np.float32)
    )
    ds2 = (
        xr.open_dataset(
            "%s/era5_%s_sfc.nc" % (input_data_dir, date2[:7]), engine=engine
        )
        .sel(time=date2)
        .astype(np.float32)
    )
    ds_sfc = ds1 - ds2

    Etot = tke(ds_pl, ds_sfc)
    p_pl = namelist["alpha"] * ds_pl / Etot
    p_sfc = namelist["alpha"] * ds_sfc / Etot

    perturbed_pl = (
        input_pl + p_pl[namelist["pl_vars"]].astype(np.float32).to_array().to_numpy()
    )
    perturbed_sfc = (
        input_sfc + p_sfc[namelist["sfc_vars"]].astype(np.float32).to_array().to_numpy()
    )

    return perturbed_pl, perturbed_sfc


def random_noise_perturbation(input_pl:np.ndarray, input_sfc:np.ndarray, lev:np.ndarray)-> tuple:
    """ Creates random noise perturbations, based on Gaussian or Perlin noise.

    Args:
        input_pl (np.ndarray): Pressure level data.
        input_sfc (np.ndarray): Surface data.
        lev (np.ndarray): Pressure levels.

    Returns:
        tuple: Perturbed pressure level data and surface data.
    """
    stds_pl, stds_sfc = get_stds()
    n_lev = len(lev)
    noise_type = namelist["noise_type"]
    noise_scale = namelist["noise_scale"]

    if noise_type == "gauss":
        noise_pl = (
            noise_scale
            * np.random.normal(size=input_pl.shape)
            * stds_pl.reshape(-1, n_lev, 1, 1)
        )
        noise_sfc = (
            noise_scale
            * np.random.normal(size=input_sfc.shape)
            * stds_sfc.reshape(-1, 1, 1)
        )
    elif noise_type == "perlin":
        noise_pl = (
            get_perlin_noise_2d(n_lev * input_pl.shape[0]).reshape(input_pl.shape)
            * stds_pl.reshape(-1, n_lev, 1, 1)
            * noise_scale
        )
        noise_sfc = (
            get_perlin_noise_2d(input_sfc.shape[0])
            * stds_sfc.reshape(-1, 1, 1)
            * noise_scale
        )

    perturbed_pl = input_pl + noise_pl
    perturbed_sfc = input_sfc + noise_sfc

    return perturbed_pl.astype(np.float32), perturbed_sfc.astype(np.float32)


def calculate_pf(input_pl:np.ndarray, input_sfc:np.ndarray, lev:np.ndarray, pert_number:int, file: Dataset, engine="netcdf4")-> None:
    """ Calculates the perturbed forecast for a given input with a specified method and saves the output to a specified file.

    Args:
        input_pl (np.ndarray): Pressure level data.
        input_sfc (np.ndarray): Surface data.
        lev (np.ndarray): Pressure levels.
        pert_number (int): Index of the perturbation.
        file (Dataset): The file to which the output is saved.
        engine (str, optional): Engine for reading files. Defaults to "netcdf4".
    """
    if namelist["ens_mode"] == "rfp":
        perturbed_pl, perturbed_sfc = random_field_perturbation(
            input_pl, input_sfc, engine
        )
    elif namelist["ens_mode"] == "noise":
        perturbed_pl, perturbed_sfc = random_noise_perturbation(
            input_pl, input_sfc, lev
        )
    elif namelist["ens_mode"] == "ecmwf":
        perturbed_pl, perturbed_sfc = input_pl, input_sfc

    # Run perturbed inference
    inference(perturbed_pl, perturbed_sfc, pert_number, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some information.")
    parser.add_argument(
        "date", type=str, help="forecast initialization date in yyyy-mm-dd"
    )
    parser.add_argument("input_data_dir", type=str)
    parser.add_argument("output_data_dir", type=str)
    parser.add_argument("filename", type=str)
    args = parser.parse_args()

    date = args.date
    filename = args.filename
    # The directory of your input and output data
    input_data_dir = args.input_data_dir
    output_data_dir = args.output_data_dir

    # Time model
    start_time = datetime.datetime.now()

    # Model initialization
    model_24 = onnx.load("pangu_weather_24.onnx")
    model_6 = onnx.load("pangu_weather_6.onnx")

    # Set the behavier of onnxruntime
    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_mem_reuse = False
    # Increase the number for faster inference and more memory consumption
    options.intra_op_num_threads = 1

    # Set the behavier of cuda provider
    cuda_provider_options = {
        "arena_extend_strategy": "kSameAsRequested",
    }

    # Initialize onnxruntime session for Pangu-Weather Models
    ort_session_24 = ort.InferenceSession(
        "pangu_weather_24.onnx",
        sess_options=options,
        providers=[("CUDAExecutionProvider", cuda_provider_options)],
    )
    ort_session_6 = ort.InferenceSession(
        "pangu_weather_6.onnx",
        sess_options=options,
        providers=[("CUDAExecutionProvider", cuda_provider_options)],
    )

    # List of leadtimes in hours since initialization
    leadtimes = [t for t in range(0, namelist["lt"] + 1, namelist["dt"])]

    # Create h5 file for forecast
    filename_out = os.path.join(output_data_dir, "fc_%s.nc" % (date))
    # Check if file already exists
    if os.path.exists(filename_out):
        os.remove(filename_out)
    # Create file
    f = Dataset(filename_out, "a")

    # Create dimensions
    f.createDimension("pert", None)
    f.createDimension("var", None)
    f.createDimension("lead_time", len(leadtimes))
    f.createDimension("lat", len(namelist["lat_subset"]))
    f.createDimension("lon", len(namelist["lon_subset"]))

    # Create variables
    time = f.createVariable("lead_time", "i4", ("lead_time",))
    forecast = f.createVariable(
        "forecast", "f4", ("pert", "var", "lead_time", "lat", "lon")
    )
    time[:] = leadtimes

    # Different treatment for ecmwf variables
    if namelist["ens_mode"] == "ecmwf":
        print("Now calculating ensembles using", namelist["ens_mode"])
        # load pressure level data
        pl = xr.open_dataset(
            "%s/%s" % (input_data_dir, filename), engine=namelist["engine"]
        ).astype(np.float32)
        lev = pl["plev"].values
        # Load the surface data
        sfc = xr.open_dataset(
            "%s/%s" % (input_data_dir, filename.replace("pl", "sfc")),
            engine=namelist["engine"],
        ).astype(np.float32)
        for e in range(namelist["ne"]):
            # Choose perturbation
            input_pl = pl[namelist["pl_vars"]].isel(time=e).to_array().to_numpy()
            input_sfc = sfc[namelist["sfc_vars"]].isel(time=e).to_array().to_numpy()

            # Calculate perturbation
            calculate_pf(
                input_pl,
                input_sfc,
                lev,
                pert_number=e,
                file=forecast,
                engine=namelist["engine"],
            )

    else:
        # Read input
        input_pl, input_sfc, lev, lat, lon = read_input(
            filename, date, engine=namelist["engine"]
        )

        # Calculate control forecast
        print("Now calculating control forecast")
        calculate_cf(input_pl, input_sfc, file_index=0, file=forecast)

        if namelist["ne"] > 0:
            print("Now calculating ensembles using ", namelist["ens_mode"])

            # Loop over number of ensemble members ne
            for e in range(1, namelist["ne"] + 1):
                calculate_pf(
                    input_pl,
                    input_sfc,
                    lev,
                    pert_number=e,
                    file=forecast,
                    engine=namelist["engine"],
                )

    f.close()
    time_elapsed = datetime.datetime.now() - start_time
    print("Time elapsed for date {}: {}".format(date, time_elapsed))
