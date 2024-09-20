import xarray as xr

def load_data_all_train_proc_norm(chunks=None):
    """
    Load all variable train processed and normed data and format dimensions
    Args:
        chunks (dict): Chunk sizes to use for the Dask arrays.
    Returns:
        list: processsed dataset of train variables from 2018 - 2021 
              order_var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
    """
    # Load all data with chunks
    dat_train_u10 = xr.open_dataset("../drn/data/train_test/u10_train.h5", chunks=chunks)
    dat_train_v10 = xr.open_dataset("../drn/data/train_test/v10_train.h5", chunks=chunks)
    dat_train_t2m = xr.open_dataset("../drn/data/train_test/t2m_train.h5", chunks=chunks)
    dat_train_t850 = xr.open_dataset("../drn/data/train_test/t850_train.h5", chunks=chunks)
    dat_train_z500 = xr.open_dataset("../drn/data/train_test/z500_train.h5", chunks=chunks)
    #dat_train_ws10 = xr.open_dataset("../drn/data/Mean_ens_std/ws10_train.h5", chunks=chunks)
    
    dat_train_all = [
        dat_train_u10,
        dat_train_v10,
        dat_train_t2m,
        dat_train_t850,
        dat_train_z500,
        #dat_train_ws10
    ]

    var_dict = {
        "phony_dim_0": "forecast_date",
        "phony_dim_1": "lead_time",
        "phony_dim_2": "lat",
        "phony_dim_3": "lon",
        "phony_dim_4": "mean_std",
    }

    dat_all = []

    for i in range(len(dat_train_all)):
        dat_all.append(dat_train_all[i].rename_dims(var_dict))

    return dat_all
def load_data_t2m_ws10_train_proc_norm():
    '''
    load t2m and ws10 train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    dat_all_important = [dat_all[2],dat_all[5]]
    return dat_all_important

def load_data_u10_train_proc_norm():
    '''
    load u10 train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    return dat_all[0]


def load_data_v10_train_proc_norm():
    '''
    load v10 train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    return dat_all[1]


def load_data_t2m_train_proc_norm():
    '''
    load t2m train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    return dat_all[2]


def load_data_t850_train_proc_norm():
    '''
    load t850 train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    return dat_all[3]


def load_data_z500_train_proc_norm():
    '''
    load z500 train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    return dat_all[4]

def load_data_ws10_train_proc_norm():
    '''
    load ws10 train data processed and normed
    '''
    dat_all = load_data_all_train_proc_norm()
    return dat_all[5]



def load_data_all_test_proc_norm(chunks=None):
    """
    Load all variable test processed and normed data and format dimensions
    Args:
        chunks (dict): Chunk sizes to use for the Dask arrays.
    Returns:
        list: processsed dataset of test variables from 2018 - 2021 
              order_var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]
    """
    # Load all data with chunks
    dat_test_u10 = xr.open_dataset("../drn/data/train_test/u10_test.h5", chunks=chunks)
    dat_test_v10 = xr.open_dataset("../drn/data/train_test/v10_test.h5", chunks=chunks)
    dat_test_t2m = xr.open_dataset("../drn/data/train_test/t2m_test.h5", chunks=chunks)
    dat_test_t850 = xr.open_dataset("../drn/data/train_test/t850_test.h5", chunks=chunks)
    dat_test_z500 = xr.open_dataset("../drn/data/train_test/z500_test.h5", chunks=chunks)
    dat_test_ws10 = xr.open_dataset("../drn/data/train_test/ws10_test.h5", chunks=chunks)

    dat_test_all = [
        dat_test_u10,
        dat_test_v10,
        dat_test_t2m,
        dat_test_t850,
        dat_test_z500,
        dat_test_ws10
    ]

    var_dict = {
        "phony_dim_0": "forecast_date",
        "phony_dim_1": "lead_time",
        "phony_dim_2": "lat",
        "phony_dim_3": "lon",
        "phony_dim_4": "mean_std",
    }
    var_names = ["u10", "v10", "t2m", "t850", "z500", "ws10"]

    dat_all = []

    for i in range(5):
        dat_test_all[i] = dat_test_all[i].rename_vars({var_names[i] + "test_truth": var_names[i] +"_test_truth"})
        dat_all.append(dat_test_all[i].rename_dims(var_dict))
    dat_all.append(dat_test_all[5].rename_dims(var_dict))
    return dat_all

def load_data_five_test_proc_norm(chunks=None):
    """
    Load all variable test processed and normed data and format dimensions
    Args:
        chunks (dict): Chunk sizes to use for the Dask arrays.
    Returns:
        list: processsed dataset of test variables from 2018 - 2021 
              order_var_names = ["u10", "v10", "t2m", "t850", "z500"]
    """
    # Load all data with chunks
    dat_test_u10 = xr.open_dataset("../drn/data/train_test/u10_test.h5", chunks=chunks)
    dat_test_v10 = xr.open_dataset("../drn/data/train_test/v10_test.h5", chunks=chunks)
    dat_test_t2m = xr.open_dataset("../drn/data/train_test/t2m_test.h5", chunks=chunks)
    dat_test_t850 = xr.open_dataset("../drn/data/train_test/t850_test.h5", chunks=chunks)
    dat_test_z500 = xr.open_dataset("../drn/data/train_test/z500_test.h5", chunks=chunks)

    dat_test_all = [
        dat_test_u10,
        dat_test_v10,
        dat_test_t2m,
        dat_test_t850,
        dat_test_z500,
    ]

    var_dict = {
        "phony_dim_0": "forecast_date",
        "phony_dim_1": "lead_time",
        "phony_dim_2": "lat",
        "phony_dim_3": "lon",
        "phony_dim_4": "mean_std",
    }
    var_names = ["u10", "v10", "t2m", "t850", "z500"]

    dat_all = []

    for i in range(5):
        dat_test_all[i] = dat_test_all[i].rename_vars({var_names[i] + "test_truth": var_names[i] +"_test_truth"})
        dat_all.append(dat_test_all[i].rename_dims(var_dict))
    return dat_all

def load_data_t2m_ws10_test_proc_norm():
    '''
    load t2m and ws10 test data processed and normed
    '''
    dat_all = load_data_all_test_proc_norm()
    dat_all_important = [dat_all[2],dat_all[5]]
    return dat_all_important


def load_data_u10_test_proc_norm():
    '''
    load u10 test data processed and normed
    '''
    dat_all = load_data_all_test_proc_norm()
    return dat_all[0]


def load_data_v10_test_proc_norm():
    '''
    load v10 test data processed and normed
    '''
    dat_all = load_data_all_test_proc_norm()
    return dat_all[1]


def load_data_t2m_test_proc_norm():
    '''
    load t2m test data processed and normed
    '''
    dat_all = load_data_all_test_proc_norm()
    return dat_all[2]


def load_data_t850_test_proc_norm():
    '''
    load t850 test data processed and normed
    '''
    dat_all = load_data_all_test_proc_norm()
    return dat_all[3]


def load_data_z500_test_proc_norm():
    '''
    load z500 test data processed and normed
    '''
    dat_all = load_data_all_test_proc_norm()
    return dat_all[4]

def load_data_ws10_test_proc_norm():
    '''
    load ws10 test data processed and normed
    '''
    dat_all = load_data_all_test_proc_norm()
    return dat_all[5]