#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=25:00:00
#SBATCH --mem-per-gpu=16000

year="2022"
input_data_dir=""
output_data_dir=""

for m in 8 9; do
    for dd in 20 21 22 23 24 25 26 27 28 29 30; do
        date="${year}-0${m}-${dd}"
        hour="00"
    
        file=era5_${date:0:7}_pl.nc
    
        if [[ -e "${input_data_dir}/$file" ]]; then 
            python Pangu_era5.py ${date}${inittime} $input_data_dir $output_data_dir ${file}
        fi
    done
done
