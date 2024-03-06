# Pangu-Weather

This folder includes the implementation of different perturbation methods for the Pangu-Weather model. The code is adapted from the [official repository](https://github.com/198808xc/Pangu-Weather/tree/main) where you can find instructions on running the model, as well as downloading the weights and  installing the model.

## Installation

We created a conda environment ready to use for running the Pangu-Weather model. In order to use it create a conda environment using

```conda env create -f Pangu_GPU.yml```.

Further instructions can be found in the [conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

## Code


| File | Description |
| ---- | ----------- | 
| `stats` | Includes the normalization statistics for the weather variables.|
| `constants.py` | Physical constants required for the random field perturbations.|
| `Pangu_era5.py` | File for creating forecasts using the Pangu-Weather model. |
| `Pangu_era5.sh` | An example shell script for running the model on a daily forecast basis with ERA5 data. |
| `Pangu_GPU.yml` | Yaml file, specifying the required conda environment. |
| `Pangu_namelist.py` | A list of parameters and settings used to run the model. |

## License

The trained parameters of Pangu-Weather were made available under the terms of the [BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/). 