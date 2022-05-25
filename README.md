# U-NO
This repository contains code accompaning the paper: [U-NO: U-shaped Neural Operators](https://arxiv.org/pdf/2204.11127.pdf)

## Requirements
pytorch 1.11.0

## Files
darcy_flow_main.py - Script for running UNO performing 2D spatial covolution for solving Darcy Flow equation. 

ns_uno2d_main.py - Script for UNO with recurrent strcuture in time for Navier-Stocks equation.

ns_uno3d_main.py - Script for UNO performing 3D spatio-temporal convolution for Navier-Stocks equation.

navier_stokes_uno2d.py - UNO(2D) achitectures autogressive in time for Navier-Stocks equation.

navier_stokes_uno3d.py - UNO(3D) achitectures performing 3D convolution for Navier-Stocks equation.

darcy_flow_uno2d.py - UNO achitectures for solving Darcy Flow equation.

ns_train_2d.py - training routine for autogressive UNO in time for Navier-Stocks equation

ns_train_3d.py - training routine for UNO (3D) for Navier-Stocks equation.

train_darcy.py - training routine for Darcy flow equations.


# Data

Link to two files containing 2000 simulations of Darcy Flow equation:
[Google Drive Link](https://drive.google.com/drive/folders/1y6j5sL4QrpKTMrlVAyN7bUlt785oQtOm?usp=sharing)

The **Data Generator** folder contains script for generating simulation of  Darcy Flow and Navier-Stocks equation.
