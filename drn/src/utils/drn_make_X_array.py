import numpy as np
import xarray as xr
import data.processed.load_data_processed_denormed as ldpd


def make_X_array(X_array_all_denorm, lead_time):
    """
    makes a training array with all of the variable mean and std combined, including embedding array while scaling them with their max values
Args:
    X_array_all_denorm (nested_list): 6x31, X values for each variable and lead_time
    lead_time (int): the lead_time for which to construct the training array (0 - 30)
Returns:
    result (array): Return array with all X_means and stds combined, order: variable_i_mean, variables_i_std, for i = 1,...,6
    embedding (array): embedding array with all embeddings in same order as results
    
    """
    X_train_lead_denorm_list = []
    mean_max = ldpd.load_max_mean_std_values_denorm()
    mean_min = ldpd.load_min_mean_std_values_denorm()
    mean_std_max = [mean_max]
    mean_std_min = [mean_min]
    for var in range(5):
        for mean_std in range(1):
            X_train_part, embedding = flatten_with_grid_ids(
                X_array_all_denorm[var][lead_time].isel(mean_std=mean_std)
            )
            X_train_part = (X_train_part - mean_std_min[mean_std][var, lead_time]) / (mean_std_max[mean_std][var, lead_time] - mean_std_min[mean_std][var, lead_time])
            X_train_lead_denorm_list.append(X_train_part)

    # get length of individual arrays and total count
    length = len(X_train_lead_denorm_list[0])
    n = len(X_train_lead_denorm_list)

    # initialize an empty array of shape (length, n)
    result = np.empty((length, n))

    # fill the result array
    for i, arr in enumerate(X_train_lead_denorm_list):
        result[:, i] = arr

    return result, embedding


def flatten_with_grid_ids(da):
    """
    Flatten an xarray DataArray and generate corresponding grid point IDs.
    
    Args:
        da (xarray.DataArray): The DataArray to flatten.
        
    Returns:
        A tuple (flattened_values, grid_ids), where:
            - flattened_values (numpy.ndarray): A 1D array with all values from the DataArray.
            - grid_ids (numpy.ndarray): A 1D array with the corresponding grid point ID for each value.
    """
    # Get the shapes of the 'lat' and 'lon' dimensions
    lat_shape = da.sizes["lat"]
    lon_shape = da.sizes["lon"]

    # Generate a 2D array with the grid point ID for each (lat, lon) pair
    grid_id_2d = np.arange(lat_shape * lon_shape).reshape(lat_shape, lon_shape)

    # Repeat the 2D grid ID array along the other dimensions to match the shape of the DataArray
    grid_id_nd = np.repeat(grid_id_2d[None, :, :], da.sizes["forecast_date"], axis=0)

    # Flatten both the DataArray values and the grid ID array
    flattened_values = da.values.flatten()
    grid_ids = grid_id_nd.flatten()

    return flattened_values, grid_ids