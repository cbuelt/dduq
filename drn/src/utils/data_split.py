def split_var_lead(dat):
    """
    Data split into lead times and variables for all X values data, excludes first lead_time as it is aquivalent to forecast_date
Args:
    dat(list): list of all 6 variables and their data including predictions and truth
    
    
Returns: 
    list: Nested list of lenght 6x31 with X values for all 6 variables and all 31 lead_times
    list: Nested list of lenght 6x31 with y values for all 6 variables and all 31 lead_times
    """
    var_names = ["u10", "v10", "t2m", "t850", "z500"]
    dat_X_lead_u10 = []  # list(31) of u10 with different lead times
    dat_X_lead_v10 = []
    dat_X_lead_t2m = []
    dat_X_lead_t850 = []
    dat_X_lead_z500 = []
    dat_X_lead_all = [
        dat_X_lead_u10,
        dat_X_lead_v10,
        dat_X_lead_t2m,
        dat_X_lead_t850,
        dat_X_lead_z500
    ]  # List of all 5 X - variables and their lead times
    dat_y_lead_u10 = []  # list(31) of u10 ground truth values with different lead_times
    dat_y_lead_v10 = []
    dat_y_lead_t2m = []
    dat_y_lead_t850 = []
    dat_y_lead_z500 = []
    dat_y_lead_all = [
        dat_y_lead_u10,
        dat_y_lead_v10,
        dat_y_lead_t2m,
        dat_y_lead_t850,
        dat_y_lead_z500
    ]
    for lead in range(1, 32):
        for var in range(5):
            dat_X_lead_all[var].append(
                dat[var][list(dat[var].data_vars.keys())[0]].isel(lead_time=lead)
            )
            dat_y_lead_all[var].append(
                dat[var][list(dat[var].data_vars.keys())[1]].isel(lead_time=lead)
            )
    return dat_X_lead_all, dat_y_lead_all


def split_var_lead_one(dat):
    """
    Return 31 datasets one for each relevant lead_time
    """
    dat_split_X = []
    dat_split_y = []
    for lead_time in range(1, 32):
        dat_split_X.append(dat[list(dat.data_vars.keys())[0]].isel(lead_time=lead_time))
        dat_split_y.append(dat[list(dat.data_vars.keys())[1]].isel(lead_time=lead_time))
    return dat_split_X, dat_split_y