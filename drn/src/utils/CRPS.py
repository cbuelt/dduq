# Basics
import numpy as np
import keras.backend as K
import tensorflow as tf
#Norms
from scipy.stats import norm
#Helpful
from tqdm import tqdm


def crps_cost_function(y_true, y_pred, theano=False):
    """Compute the CRPS cost function for a normal distribution defined by
    the mean and standard deviation.

    Code inspired by Kai Polsterer (HITS).

    Args:
        y_true: True values
        y_pred: Tensor containing predictions: [mean, std]
        theano: Set to true if using this with pure theano.

    Returns:
        mean_crps: Scalar with mean CRPS over batch
    """

    # Split input
    mu = y_pred[:, 0]
    sigma = y_pred[:, 1]
    # Ugly workaround for different tensor allocation in keras and theano
    if not theano:
        y_true = y_true[:, 0]   # Need to also get rid of axis 1 to match!

    # To stop sigma from becoming negative we first have to convert it the the variance and then take the square root again. 
    var = K.square(sigma)
    # The following three variables are just for convenience
    loc = (y_true - mu) / K.sqrt(var)
    phi = 1.0 / np.sqrt(2.0 * np.pi) * K.exp(-K.square(loc) / 2.0)
    Phi = 0.5 * (1.0 + tf.math.erf(loc / np.sqrt(2.0)))
    # First we will compute the crps for each input/target pair
    crps =  K.sqrt(var) * (loc * (2. * Phi - 1.) + 2 * phi - 1. / np.sqrt(np.pi))
    # Then we take the mean. The cost is now a scalar
    
    return K.mean(crps)


def crps_normal(mu, sigma, y):
    """
    Compute CRPS for a Gaussian distribution:
    Input: 
        param mu: Array or pd.Series or XArray of ensemble means
        param sigma: Array or pd.Series or XArray  of ensemble stds
        param y: Array or pd.Series or XArray  ground truth values
        
    Output:
        crps (numpy): Continuous ranked probability score for a Gaussian distribution
    """
    # Make sure sigma is positive
    sigma = np.abs(sigma)
    loc = (y - mu) / sigma
    crps = sigma * (loc * (2 * norm.cdf(loc) - 1) + 2 * norm.pdf(loc) - 1. / np.sqrt(np.pi))
    
    return crps

def crps_cost_function_trunc(y_true, y_pred, theano=False):
    '''
    Crps cost function truncated for normal distributions
    '''
    mu = K.abs(y_pred[:, 0])
    sigma = K.abs(y_pred[:, 1])
    
    if not theano:
        y_true = y_true[:, 0]   # Need to also get rid of axis 1 to match!
        
    var = K.square(sigma)
    loc = (y_true - mu) / K.sqrt(var)
    
    phi = 1.0 / np.sqrt(2.0 * np.pi) * K.exp(-K.square(loc) / 2.0)
    
    Phi_ms = 0.5 * (1.0 + tf.math.erf(mu/sigma / np.sqrt(2.0)))
    Phi = 0.5 * (1.0 + tf.math.erf(loc / np.sqrt(2.0)))
    Phi_2ms = 0.5 * (1.0 + tf.math.erf(np.sqrt(2)*mu/sigma / np.sqrt(2.0)))
    
    crps = K.sqrt(var) / K.square( Phi_ms ) * (
            loc * Phi_ms * (2.0 * Phi + Phi_ms - 2.0)
            + 2.0 * phi * Phi_ms - 1.0 / np.sqrt(np.pi) * Phi_2ms
        )
    return K.mean(crps)


def crps_trunc(mu, sigma, y):
    '''
        Compute CRPS for a truncated Gaussian distribution:
    Input: 
        param mu: Array or pd.Series or XArray of ensemble means
        param sigma: Array or pd.Series or XArray  of ensemble stds
        param y: Array or pd.Series or XArray  ground truth values
        
    Output:
        crps(numpy): Continuous ranked probability score for a Gaussian distribution
    
    '''
    y_pred = np.stack((mu, sigma), axis=1)
    y_true = y

    mu = K.abs(y_pred[:, 0])
    sigma = K.abs(y_pred[:, 1])
        
    var = K.square(sigma)
    loc = (y_true - mu) / K.sqrt(var)
    
    phi = 1.0 / np.sqrt(2.0 * np.pi) * K.exp(-K.square(loc) / 2.0)
    
    Phi_ms = 0.5 * (1.0 + tf.math.erf(mu/sigma / np.sqrt(2.0)))
    Phi = 0.5 * (1.0 + tf.math.erf(loc / np.sqrt(2.0)))
    Phi_2ms = 0.5 * (1.0 + tf.math.erf(np.sqrt(2)*mu/sigma / np.sqrt(2.0)))
    
    crps = K.sqrt(var) / K.square( Phi_ms ) * (
            loc * Phi_ms * (2.0 * Phi + Phi_ms - 2.0)
            + 2.0 * phi * Phi_ms - 1.0 / np.sqrt(np.pi) * Phi_2ms
        )
    return crps.numpy()

def crps_var_lead(X_test_lead_all, y_test_var_lead_all):
    """
    Calculate CRPS Baseline score of dataset for all variables and all lead_times
Args: 
    X_test_lead_all (list): List of xarray dataArrays X-values for each variable
    y_test_lead_all (list): List of xarray dataArrays y-values for each variable
Returns:
    nested_list: nested list with var-lead_time(5x31) with all crps values calculated.
    """
    crps_baseline_all = [[], [], [], [], []]
    for var in tqdm(range(5)):
        for lead_time in range(31):
            # CRPS distribution should be dependent on variable
            if var in [4]:
                crps_baseline = crps_trunc(
                mu=X_test_lead_all[var][lead_time].isel(mean_std=0).values,
                sigma=X_test_lead_all[var][lead_time].isel(mean_std=1).values,
                y=y_test_var_lead_all[var][lead_time].values,
            )
            else:
                crps_baseline = crps_normal(
                mu=X_test_lead_all[var][lead_time].isel(mean_std=0).values,
                sigma=X_test_lead_all[var][lead_time].isel(mean_std=1).values,
                y=y_test_var_lead_all[var][lead_time].values,
            )
            
            crps_baseline_all[var].append(crps_baseline)
    return crps_baseline_all

def crps_var_lead_preds(Mean_std_predictions, y_test_var_lead_all):
    """
    Calculate CRPS of dataset for all variables and all lead_times
Args: 
    Mean_std_predictions (list): List of xarray dataArrays X-values for each variable and lead_time
    y_test_lead_all (list): List of xarray dataArrays y-values for each variable and lead_time
Returns:
    nested_list: nested list with var-lead_time(5x31) with all crps values calculated.
    """
    crps_baseline_all_preds = [[], [], [], [], []]
    for var in range(5):
        for lead_time in range(31):
            if var in [4]:
                crps_baseline = crps_trunc(
                    mu=Mean_std_predictions[var][lead_time][:, 0].flatten(),
                    sigma=Mean_std_predictions[var][lead_time][:, 1].flatten(),
                    y=y_test_var_lead_all[var][lead_time].values,
                )
            else:
                crps_baseline = crps_normal(
                    mu=Mean_std_predictions[var][lead_time][:, 0].flatten(),
                    sigma=Mean_std_predictions[var][lead_time][:, 1].flatten(),
                    y=y_test_var_lead_all[var][lead_time].values,
                )
            crps_baseline_all_preds[var].append(crps_baseline)
    return crps_baseline_all_preds