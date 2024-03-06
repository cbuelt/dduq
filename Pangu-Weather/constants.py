constants = dict()
# Reference temperature
constants['T_r'] = 300.
# Reference pressure
constants['P_r'] = 800.
# gravitational acceleration
constants['g'] = 9.80665
# Molar gas constant
constants['R'] = 8.314462618
# dry air molecular weight
constants['Md'] = 28.96546e-3
# dry air gas constant
constants['Rd'] = (constants['R'] \
    / constants['Md'])
# dry air specific heat ratio
constants['dry_air_spec_heat_ratio'] = 1.4
# dry air specific heat at constant pressure
constants['Cp_d'] = (constants['dry_air_spec_heat_ratio'] * constants['Rd'] \
    / (constants['dry_air_spec_heat_ratio'] - 1))
