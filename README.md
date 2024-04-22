# U-NO
<img src="https://raw.githubusercontent.com/ashiq24/UNO/web_resources/uno.png" alt="uno architecture" style="height: 250px; width:300px;"/>
This repository contains code accompanying the paper: [U-NO: U-shaped Neural Operators](https://arxiv.org/pdf/2204.11127.pdf)

**UNO_Tutorial.ipynb** - A step-by-step tutorial for using and buidling U-NO. Link to Google colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f1WYsjAgIjJRFtfQYYnZCZsxl602MMPX?usp=sharing)

**U-NO** is now available on [**Neural Operator**](https://github.com/neuraloperator/neuraloperator) library. [Quick Start](https://github.com/neuraloperator/neuraloperator/blob/main/examples/plot_UNO_darcy.py)
## Requirements
PyTorch 1.11.0

## Files

| Files| Descriptions|
|------|-------------|
|integral_operators.py | Contains codes for Non-linear integral operators for 1D, 2D, and 3D functions.|
|UNO_Tutorial.ipynb| A tutorial on using the integral operators and U-NO.|
|**Darcy Flow**|
|darcy_flow_main.py | Script for loading data, training, and evaluating training UNO performing 2D spatial convolution for solving Darcy Flow equation.|
|darcy_flow_uno2d.py | UNO architectures for solving Darcy Flow equation.|
|train_darcy.py | Training routine for Darcy flow equations.|
|data_load_darcy.py| Function to load Darct-flow data.|
|**Navier–Stokes**|
|data_load_navier_stocks.py| Function to load Navier–Stokes data generated by data generator prodived|
|ns_uno2d_main.py | Script for loading data, training, and evaluating the UNO (2D) autoregressive in time for Navier–Stokes equation.|
|ns_train_2d.py | Training function for UNO(2D) in time for Navier–Stokes equation|
|navier_stokes_uno2d.py | UNO(2D) architecture in time for Navier–Stokes equation.|
|ns_uno3d_main.py | Script for loading data,training and evaluating the UNO(3D) performing 3D (spatio-temporal) convolution for Navier–Stokes equation.|
|navier_stokes_uno3d.py | UNO(3D) achitectures performing 3D convolution for Navier–Stokes equation.|
|ns_train_3d.py | Training function for UNO(3D) for Navier–Stokes equation.|
|**Supporting Files**|
|Data Generation| Folder contains scripts to generate data from Navier–Stokes equation and Darcy flow|
|utilities3.py| Contains supporting functions for data loading and error estimation.|


## Data

Link to two files containing 2000 simulations of Darcy Flow equation:
[Google Drive Link](https://drive.google.com/drive/folders/1y6j5sL4QrpKTMrlVAyN7bUlt785oQtOm?usp=sharing)

The **Data Generator** folder contains script for generating simulation of  Darcy Flow and Navier-Stocks equation.
