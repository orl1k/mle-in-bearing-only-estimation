# MLE in bearings-only parameter estimation

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/orl1k/mle-in-bearing-only-estimation/HEAD)

A Binder-compatible repo with a requirements.txt file.  
Access this Binder at the following URL:

https://mybinder.org/v2/gh/orl1k/mle-in-bearing-only-estimation/HEAD

Project structure:  
![Project structure](https://github.com/orl1k/mle-in-bearing-only-estimation/blob/main/images/project_backend_structure.jpg)

# Problem Statement:
Find parameters of uniform and rectilinear object's motion given physical system, maneuvering observer  
Common object trajectory example:  
![Object trajectory example](https://github.com/orl1k/mle-in-bearing-only-estimation/blob/main/images/trajectory_example.gif)  

Observer's data is formed of bearing (angle) measurements with normal distributed error  
Example of real data vs noised data:  
![Example of pseudo-real data](https://github.com/orl1k/mle-in-bearing-only-estimation/blob/main/images/measurements_vs_real_data.png)

# Implemented 4 methods to solve the problem:  
- maximum likelihood using lev-mar algorithm and embedded in scipy
- dynamic maximum likelihood
- n bearings

Convergence of maximum likelihood lev-mar:  
![MLE lev-mar convergence](https://github.com/orl1k/mle-in-bearing-only-estimation/blob/main/images/mle_levmar_convergence.gif)

Algorithm comparsion:  
![Algorithm comparsion](https://github.com/orl1k/mle-in-bearing-only-estimation/blob/main/images/n_vs_mle.png)

For more results check folder tests.  
For more information check research paper:

https://github.com/orl1k/mle-in-bearing-only-estimation/blob/main/images/research.pdf
