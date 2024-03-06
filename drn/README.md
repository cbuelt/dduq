# DRN

This folder includes all scripts and methods for applying the Distributional Regression Network (DRN) to the forecast of the Pangu-Weather model. As the model requires a different data structure, the forecasts from need to be transformed first.

## Code


|Folder|Subfolder| Description|
|----|--------|----------|
|`data`|`processed`| Code for loading the pre-processed data.|
|`src`|`data`| Code for creating training/testing data from data-driven model forecasts, as well as creating summary statistics|
||`DRN`|Code for training and evaluating the DRN model, based on pre-specified hyperparameters.|
||`utils`|Utility functions.|

