import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')
import data.processed.load_data_processed_denormed as ldpd
from src.utils.data_split import *
from tqdm import tqdm

def make_mean_std_min(dat_var_lead_all):
    path = "../drn/data/mean_std_min_values/"
    mean_min = np.zeros([5, 31])
  #  std_min = np.zeros([6, 31])
    for var in tqdm(range(5)):
        for lead_time in range(31):
            mean_min[var, lead_time] = (
                dat_var_lead_all[var][lead_time].isel(mean_std=0).min().values
            )
       #     std_min[var, lead_time] = (
     #           dat_var_lead_all[var][lead_time].isel(mean_std=1).min().values
       #     )
    np.save("../drn/data/mean_std_min_values/mean_min.npy", mean_min)
    #np.save("../DRN/data/mean_std_min_values/denorm/std_min.npy", std_min)
    
if __name__ == "__main__":
    # Call the main function
    dat_train_denorm = ldpd.load_data_all_train_proc_denorm()
    dat_X_train_lead_all_denorm, dat_y_train_lead_all_denorm = split_var_lead(
    dat_train_denorm
)
    make_mean_std_min(dat_X_train_lead_all_denorm)
