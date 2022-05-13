# U-NO
This repository contains code accompaning the paper: [U-NO: U-shaped Neural Operators](https://arxiv.org/pdf/2204.11127.pdf)
## Requirements
pytorch 1.11.0
## Files
darcy_uno2d.py - Script for  UNO performing 2D spatial covolution for solving Darcy Flow equation. (Section 4.2, [paper](https://arxiv.org/pdf/2204.11127.pdf))

ns_uno2d_time.py - Script for UNO with recurrent strcuture in time for Navier-Stocks equation. (Section 4.3, [paper](https://arxiv.org/pdf/2204.11127.pdf))

ns_uno3d.py - Script for UNO performing 3D spatio-temporal convolution for Navier-Stocks equation. (Section 4.3, Appendix C, [paper](https://arxiv.org/pdf/2204.11127.pdf))

UNO2D.py - UNO achitectures autogressive in time for Navier-Stocks equation.

UNO3D.py - UNO achitectures performing 3D convolution for Navier-Stocks equation.

darcy_model.py - UNO achitectures for solving Darcy Flow equation.


