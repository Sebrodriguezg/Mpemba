# Code and Data for Quantum Mpemba Effect in Random Circuits

The two vector simulators are written in Fortran. The tensor network simulator is written using the [ITensor](https://itensor.org/) library API in C++. 
A sample of the code in the `ITensor` library for julia is presented in a notebook for $q=1$. 

The replica transfer gates are precomputed following Ref. [1,2].

For question and inquiries, please contact [us](mailto:turkeshi@thp.uni-koeln.de).

## References

1. T. Rakovszky, F. Pollmann, and C. W. von Keyserlingk, *Diffusive Hydrodynamics of Out-of-Time-Ordered Correlators with Charge Conservation*, [Phys. Rev. X 8, 031058 (2018)](https://doi.org/10.1103/PhysRevX.8.031058). 

2. T. Rakovszky, F. Pollmann, and C. W. von Keyserlingk, *Sub-ballistic Growth of Rényi Entropies due to Diffusion*, [Phys. Rev. Lett. 122, 250602 (2019)](https://doi.org/10.1103/PhysRevLett.122.250602). 


