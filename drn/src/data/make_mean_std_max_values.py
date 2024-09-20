import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')
import data.processed.load_data_processed_denormed as ldpd
from src.utils.data_split import *
from tqdm import tqdm

def make_mean_std_max(dat_var_lead_all):
    path = "../drn/data/mean_std_max_values/"
    mean_max = np.zeros([5, 31])
    #std_max = np.zeros([6, 31])
    for var in tqdm(range(5)):
        for lead_time in range(31):
            mean_max[var, lead_time] = (
                dat_var_lead_all[var][lead_time].isel(mean_std=0).max().values
            )
          #  std_max[var, lead_time] = (
          #      dat_var_lead_all[var][lead_time].isel(mean_std=1).max().values
           # )
    np.save("../drn/data/mean_std_max_values/mean_max.npy", mean_max)
    #np.save("../DRN/data/mean_std_max_values/denorm/std_max.npy", std_max)
    
if __name__ == "__main__":
    # Call the main function
    dat_train_denorm = ldpd.load_data_all_train_proc_denorm()
    dat_X_train_lead_all_denorm, dat_y_train_lead_all_denorm = split_var_lead(
    dat_train_denorm
)
    make_mean_std_max(dat_X_train_lead_all_denorm)