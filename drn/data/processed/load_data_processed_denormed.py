import xarray as xr
import numpy as np

def load_data_all_train_proc_denorm(chunks=None):
    """
    Load all variable train processed and denormed data and format dimensions
    Args:
        chunks (dict): Chunk sizes to use for the Dask arrays.
    Returns:
        list: processed dataset of train variables from 2018 - 2021 
              order_var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
    """
    chunks_ws10 = None
    if chunks != None:
        chunks_ws10 = {'phony_dim_0':10}
    # Load all data with chunks
    path = "../drn/data/train_test_denorm/"
    dat_train_u10 = xr.open_dataset(path + "u10_train_denorm.h5", chunks=chunks)
    dat_train_v10 = xr.open_dataset(path + "v10_train_denorm.h5", chunks=chunks)
    dat_train_t2m = xr.open_dataset(path + "t2m_train_denorm.h5", chunks=chunks)
    dat_train_t850 = xr.open_dataset(path + "t850_train_denorm.h5", chunks=chunks)
    dat_train_z500 = xr.open_dataset(path + "z500_train_denorm.h5", chunks=chunks)
  #  dat_train_ws10 = xr.open_dataset(path + "ws10_train_denorm.h5", chunks=chunks_ws10).rename_dims({
  #      "phony_dim_0": "forecast_date",
  #      "phony_dim_1": "lead_time",
  #      "phony_dim_2": "lat",
  #      "phony_dim_3": "lon",
  #      "phony_dim_4": "mean_std",
  #  })
    
    dat_train_all = [
        dat_train_u10,
        dat_train_v10,
        dat_train_t2m,
        dat_train_t850,
        dat_train_z500,
  #      dat_train_ws10
    ]

    return dat_train_all

def load_data_all_train_val_proc_denorm():
    train_var_denormed = load_data_all_train_proc_denorm()
    train_var_denormed_new = []
    val_var_denormed_new = []
    for i in range(5):
        train_var_denormed_new.append(train_var_denormed[i].isel(forecast_date=slice(0, 1429 - 357)))
        val_var_denormed_new.append(train_var_denormed[i].isel(forecast_date=slice(1429 - 357, 1429)))
    return train_var_denormed_new, val_var_denormed_new

def load_data_t2m_ws10_train_proc_denorm():
    '''
    load t2m and ws10 train data processed and normed
    '''
    dat_all = load_data_all_train_proc_denorm()
    dat_all_important = [dat_all[2],dat_all[5]]
    return dat_all_important

def load_data_u10_train_proc_denorm():
    '''
    load u10 train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[0]


def load_data_v10_train_proc_denorm():
    '''
    load v10 train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[1]


def load_data_t2m_train_proc_denorm():
    '''
    load t2m train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[2]


def load_data_t850_train_proc_denorm():
    '''
    load t850 train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[3]


def load_data_z500_train_proc_denorm():
    '''
    load z500 train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[4]

def load_data_ws10_train_proc_denorm():
    '''
    load ws10 train data processed and denormed
    '''
    dat_all = load_data_all_train_proc_denorm()
    return dat_all[5]


def load_data_all_test_proc_denorm(chunks=None):
    """
    Load all variable test processed and denormed data and format dimensions
    Args:
        chunks (dict): Chunk sizes to use for the Dask arrays.
    Returns:
        list: processed dataset of test variables 
              order_var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
    """
    # Load all data with chunks
    path = "../drn/data/train_test_denorm/"
    dat_test_u10 = xr.open_dataset(path + "u10_test_denorm.h5", chunks=chunks)
    dat_test_v10 = xr.open_dataset(path + "v10_test_denorm.h5", chunks=chunks)
    dat_test_t2m = xr.open_dataset(path + "t2m_test_denorm.h5", chunks=chunks)
    dat_test_t850 = xr.open_dataset(path + "t850_test_denorm.h5", chunks=chunks)
    dat_test_z500 = xr.open_dataset(path + "z500_test_denorm.h5", chunks=chunks)
 #   dat_test_ws10 = xr.open_dataset(path + "ws10_test_denorm.h5", chunks=chunks).rename_dims({
 #       "phony_dim_0": "forecast_date",
 #       "phony_dim_1": "lead_time",
 #       "phony_dim_2": "lat",
 #       "phony_dim_3": "lon",
 #       "phony_dim_4": "mean_std",
 #   })
    
    dat_test_all = [
        dat_test_u10,
        dat_test_v10,
        dat_test_t2m,
        dat_test_t850,
        dat_test_z500,
 #       dat_test_ws10
    ]

    return dat_test_all

def load_data_t2m_ws10_test_proc_denorm():
    '''
    load t2m and ws10 train data processed and normed
    '''
    dat_all = load_data_all_test_proc_denorm()
    dat_all_important = [dat_all[2],dat_all[5]]
    return dat_all_important

def load_data_u10_test_proc_denorm():
    '''
    load u10 test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[0]


def load_data_v10_test_proc_denorm():
    '''
    load v10 test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[1]


def load_data_t2m_test_proc_denorm():
    '''
    load t2m test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[2]


def load_data_t850_test_proc_denorm():
    '''
    load t850 test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[3]


def load_data_z500_test_proc_denorm():
    '''
    load z500 test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[4]

def load_data_ws10_test_proc_denorm():
    '''
    load ws10 test data processed and denormed
    '''
    dat_all = load_data_all_test_proc_denorm()
    return dat_all[5]


def load_max_mean_std_values_denorm():
    '''
    load all max mean and std values to scale the denormed data with
    '''
    mean_max = np.load("../drn/data/mean_std_max_values/mean_max.npy")
    #std_max = np.load("/Data/Delong_BA_Data/mean_std_max_values/denorm/std_max.npy")
    return mean_max#, std_max

def load_min_mean_std_values_denorm():
    '''
    load all min mean and std values to scale the denormed data with
    '''
    mean_min = np.load("../drn/data/mean_std_min_values/mean_min.npy")
    #std_min = np.load("/pfs/work7/workspace/scratch/vt0186-fourcastnet/DRN/data/mean_std_min_values/denorm/std_min.npy")
    return mean_min#, std_min







