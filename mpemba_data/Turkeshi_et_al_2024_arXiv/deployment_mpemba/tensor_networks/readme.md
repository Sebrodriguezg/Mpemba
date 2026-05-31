# Tensor network code

The implementation uses the [ITensor](https://itensor.org/) library API in C++. 
Download the library, compile it and link it. 

The replica transfer gates are precomputed following Ref. [1,2].

## References

1. T. Rakovszky, F. Pollmann, and C. W. von Keyserlingk, *Diffusive Hydrodynamics of Out-of-Time-Ordered Correlators with Charge Conservation*, [Phys. Rev. X 8, 031058 (2018)](https://doi.org/10.1103/PhysRevX.8.031058). 

2. T. Rakovszky, F. Pollmann, and C. W. von Keyserlingk, *Sub-ballistic Growth of Rényi Entropies due to Diffusion*, [Phys. Rev. Lett. 122, 250602 (2019)](https://doi.org/10.1103/PhysRevLett.122.250602). 