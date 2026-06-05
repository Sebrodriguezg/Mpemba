FEM Mpemba model (1D)

This folder contains a minimal 1D finite element solver for the Fourier-convection
model used in the PCCP 2014 paper (C4CP03669G). It uses constant properties,
a bulk/skin split domain, convection, and Robin radiation boundaries.

Model
- rho*cp * dT/dt = k * d2T/dx2 - rho*cp * v * dT/dx
- x in [-l1, l2], interface at x=0
- continuity of T and k dT/dx at x=0
- boundary: -k dT/dx = h (T - Tf)

Defaults (approx paper settings)
- l1=10 mm, l2=1 mm
- rho_s/rho_b=0.75, alpha_s/alpha_b=1.48
- v=1e-4 m/s
- h1/k_b=30, h2 = h1 * h2_h1 (extra radiation factor)
- Tf=0 C, Ti list = 20,40,60,80 C

Run
- python fem_mpemba.py

Outputs
- fourier/out/relaxation.png and relaxation.csv
- fourier/out/delta.png and delta.csv

Tuning
- Edit Params in fem_mpemba.py to match other conditions or figure panels.
- You can set upwind=False to remove artificial diffusion.

Notes
- This is a constant-property baseline. The paper uses T-dependent k, rho, Cp.
- If you want a closer match, we can add those functions next.
