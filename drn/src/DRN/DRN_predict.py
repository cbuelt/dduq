# Basics
import numpy as np
import argparse
import multiprocessing as mp

# Tensorflow and Keras
import keras.backend as K
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Xarray
import xarray as xr

# Helpful
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')

# My Methods
from src.utils.CRPS import *  # CRPS metrics
from src.utils.data_split import *  # Splitting data into X and y
from src.utils.drn_make_X_array import *  # Import make train array functions (make_X_array)
from src.DRN.DRN_model import *  # DRN implementation
import data.raw.load_data_raw as ldr  # Load raw data
import data.processed.load_data_processed as ldp  # Load processed data normed
import data.processed.load_data_processed_denormed as ldpd  # Load processed data denormed
from src.models.CRPS_baseline.CRPS_load import *  # Load CRPS scores


def DRN_predict_evaluate(var_num, lead_time):
    """
    This function loads a pre-trained Distributional Regression Network (DRN) model, applies it to the test data, 
    and saves the predictions.

    Args:
        var_num (int): Variable number between 0-5, used to select the variable. The variables are represented as:
            0: "u10"
            1: "v10"
            2: "t2m"
            3: "t850"
            4: "z500"
        lead_time (int): The lead time between 0-31 for the variable. 

    Returns:
        None. The function saves the DRN model's predictions into a .npy file for each variable at the specified lead time.
    """
    
    # Define the names of the variables
    var_names = ["u10", "v10", "t2m", "t850", "z500"]
    
    # Load the trained model to use:
    path = f'../DRN/data/models/DRN/DRN_{var_names[var_num]}_lead_time_{lead_time}_denormed.h5'
    DRN_model = tf.keras.models.load_model(path, custom_objects={
                "crps_cost_function": crps_cost_function,
                "crps_cost_function_trunc": crps_cost_function_trunc,
            })
    
    # Load all test data of each variable
    test_var_denormed = (
        ldpd.load_data_all_test_proc_denorm()
    )
    
    # Split the loaded data into features (X) and target (y)
    dat_X_test_lead_all_denorm, dat_y_test_lead_all_denorm = split_var_lead(
        test_var_denormed
    )

    # Preprocess the features for Neural Network and scale them
    drn_X_test_lead_array, drn_embedding_test_lead_array = make_X_array(
        dat_X_test_lead_all_denorm, lead_time
    ) 

    # DRN Model predictions:
    DRN_preds = DRN_model.predict([drn_X_test_lead_array, drn_embedding_test_lead_array])
    
    # Save the predictions
    model_filename = f"../DRN/data/preds/DRN_preds/DRN_{var_names[var_num]}_lead_time_{lead_time}_preds.npy"
    np.save(model_filename, DRN_preds)

    
    
    
def main(var_num, lead_time):
    DRN_predict_evaluate(var_num, lead_time)
    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Calculate CRPS for a given variable")

    # Add the arguments
    parser.add_argument('--var_num', type=int, help='Variable number between 0 and 5')
    # Parse the arguments
    args = parser.parse_args()
    
    # Create a pool of worker processes
    n_proc = len(os.sched_getaffinity(0))
    pool = mp.Pool(n_proc)

    # Create a list to store the results
    results = []

    # Call the main function for each lead_time
    for lead_time in [3,11,27]:#range(31):
        result = pool.apply_async(main, args=(args.var_num, lead_time))
        results.append(result)
    
    # Close the pool of worker processes
    pool.close()
    
    # Call get() on each result to raise any exceptions
    for result in results:
        result.get()
    
    # Wait for all processes to finish
    pool.join()