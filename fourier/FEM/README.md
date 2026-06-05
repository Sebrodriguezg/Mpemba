# FEM Mpemba Model (1D)

This folder contains a 1D Finite Element Method (FEM) solver utilizing a Galerkin linear formulation to simulate the macroscopic thermal relaxation of water, reproducing the core physical paradigm of the Mpemba paradox presented in the PCCP 2014 paper (DOI: 10.1039/c4cp03669g).

Unlike a static linear solver, this implementation incorporates non-linear thermal dynamics and historical link memory effects.

## Model Equations

The heat transport throughout the domain is governed by the 1D non-linear Fourier diffusion equation:

$$\rho(T) C_p(T) \frac{\partial \theta}{\partial t} = \frac{\partial}{\partial x} \left( k(T, x) \frac{\partial \theta}{\partial x} \right)$$

* **Domain Split:** $x \in [-l_1, l_2]$, where the bulk-skin interface is located strictly at $x = 0$[cite: 53, 54].
* **Interface Conditions:** Strict continuity of temperature ($\theta(0^-) = \theta(0^+)$) and heat flux conservation ($\kappa_- \theta_x = \kappa_+ \theta_x$)[cite: 58, 69].
* **Boundary Conditions (Robin Radiation):** Heat dissipation at both ends follows $h_i (\theta_f - \theta) \pm \kappa_i \theta_x = 0$[cite: 58, 69].
* **Non-Linear Properties:** Water density $\rho(T)$, specific heat $C_p(T)$, and thermal conductivity $k(T)$ are updated dynamically at every time-step using empirical water property curves[cite: 72, 445].
* **Hydrogen-Bond Memory Effect:** To reproduce the paradox under the *Maximum Principle* restriction of parabolic PDEs, a history-dependent `memory_factor` is coupled to the skin's thermal diffusivity $\alpha_S$ and radiation coefficient $h_2$ based on the initial temperature $T_i$[cite: 20, 181].

## Configuration & Parameters

The default settings simulate the pure static cooling case ($v = 0$ m/s) to isolate the structural effects of surface supersolidity from convective noise[cite: 75, 416]:

* **Geometry:** Bulk length $l_1 = 9$ mm, Skin length $l_2 = 1$ mm ($N_{bulk} = 91$, $N_{skin} = 11$)[cite: 53].
* **Supersolidity Ratios:** $\rho_S/\rho_B = 0.75$, $\alpha_S/\alpha_B = 1.48$[cite: 53, 68].
* **Radiation Coefficients:** $h_1/k_B = 30.0$, $h_2/h_1 = 1.0$[cite: 48, 50].
* **Temporal Discretization:** Step size $\Delta t = 1.0$ s up to $t_{end} = 3000$ s solved via an unconditionally stable Crank-Nicolson implicit scheme ($\theta_{theta} = 0.5$)[cite: 36].
* **Thermal Range:** Evaluates the critical crossing regime comparing $T_i = 20^\circ\text{C}$ and $T_i = 30^\circ\text{C}$ against a thermal drain of $T_f = 0^\circ\text{C}$[cite: 56, 105, 106].

## Execution

To run the simulation and generate the data, execute:

```bash
python fem_mpemba.py