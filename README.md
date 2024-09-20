# Uncertainty quantification for data-driven weather models

This repository provides Python code accompanying the paper
> Bülte, C., Horat, N., Quinting, J. and Lerch S. (2024). Uncertainty quantification for data-driven weather models
> (preprint available at https://arxiv.org/abs/2403.13458)
     
The code includes methods for producing ensemble forecasts using the PanguWeather model and different initial condition methods, as described in the paper. Furthermore, two post-processing approaches are implemented and can be used. In addition, code for evaluation of the different approaches is included.

## Abstract

Artificial intelligence (AI)-based data-driven weather forecasting models have experienced rapid progress over the last years. Recent studies, with models trained on reanalysis data, achieve impressive results and demonstrate substantial improvements over state-of-the-art physics-based numerical weather prediction models across a range of variables and evaluation metrics. Beyond improved predictions, the main advantages of data-driven weather models are their substantially lower computational costs and the faster generation of forecasts, once a model has been trained. However, most efforts in data-driven weather forecasting have been limited to deterministic, point-valued predictions, making it impossible to quantify forecast uncertainties, which is crucial in research and for optimal decision making in applications. Our overarching aim is to systematically study and compare uncertainty quantification methods to generate probabilistic weather forecasts from a state-of-the-art deterministic data-driven weather model, Pangu-Weather. Specifically, we compare approaches for quantifying forecast uncertainty based on generating ensemble forecasts via perturbations to the initial conditions, with the use of statistical and machine learning methods for post-hoc uncertainty quantification. In a case study on medium-range forecasts of selected weather variables over Europe, the probabilistic forecasts obtained by using the Pangu-Weather model in concert with uncertainty quantification methods show promising results and provide improvements over ensemble forecasts from the physics-based ensemble weather model of the European Centre for Medium-Range Weather Forecasts for lead times of up to 5 days.


## Code

The code is described in the corresponding folders:

| Folder | Description |
| ---- | ----------- | 
| `DRN` | Implementation of the DRN model. |
| `EasyUQ` | Implementation of the EasyUQ model. |
| `evaluation` | Evaluation of the different forecasts. |
| `Pangu-Weather` | Creating perturbed forecasts with the Pangu-Weather model. |
| `plots` | Generating the results plots. |
| `utils` | Utility functions. |
| `wb2` | Code for using the WeatherBench2 dataset. |

## Cite
```bibtex
@misc{bülte2024uncertainty,
      title={Uncertainty quantification for data-driven weather models}, 
      author={Christopher Bülte and Nina Horat and Julian Quinting and Sebastian Lerch},
      year={2024},
      eprint={2403.13458},
      archivePrefix={arXiv},
      primaryClass={physics.ao-ph}
}
```

