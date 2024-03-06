# Description: Arguments for specifying the Pangu model setup.
# Author: Christopher Bülte
import numpy as np

namelist = dict()
# Maximum lead time
namelist['lt'] = 186
# time increment between leadtimes (should not be changed)
namelist['dt'] = 6
# number of perturbed ensemble members
namelist['ne'] = 50
# Path for files with mean and std of input variables
namelist['std_path'] = 'stats/'
# Perturbation method (rfp=random field perturbation)
namelist['ens_mode'] = 'ecmwf'
# Tuning parameter (Eq. 8 in Magnusson et al. 2009)
namelist['alpha'] = 0.5e7
# Noise scaling parameter
namelist['noise_scale'] = 0.001
namelist['noise_type'] = 'gauss'
# List of years from which rfp are created
namelist['ens_years'] = [2018, 2019, 2020, 2021]
# Input variables (has to be this order for Pangu)
namelist['sfc_vars'] = ['MSL','U10M', 'V10M', 'T2M']
namelist['pl_vars'] = ['Z', 'Q', 'T', 'U', 'V']
# Format of input data
namelist['engine'] = 'netcdf4'
# Pressure levels of output
namelist['out_levs'] = [85000.,50000.,25000.]

# Latitude subset
namelist['lat_subset'] = np.arange(60,220)
# Longitude subset
namelist['lon_subset'] = np.append(np.arange(1390, 1440), np.arange(0, 170))
# Surface output variables
namelist['sfc_out'] = [1, 2, 3]
# Pressure level output variables
namelist['pl_out'] = ['Z', 'T']
# All pressure levels
Levels : [100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 5000]
# and corresponding pressure level positions
namelist['pl_lev_out'] = {2 : [2], 0 : [5] }

