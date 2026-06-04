#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cross_validate_t3a.py  --  Valida la salida del nucleo C++ de T3a (trayectorias)
contra la ecuacion maestra EXACTA (oraculo `common.qmpe`) para N=3, donde el
Liouvilliano denso (64x64) aun cabe.

Compara la densidad de excitacion n_exc(t) producida por:
  * el binario C++ qtraj_mpi  (resultados en results/t3a_p0_*.csv)
  * la integracion densa de la ecuacion maestra

Uso:
  # 1) generar datos C++:
  (cd ../cpp/build && mpirun --oversubscribe -np 2 ./qtraj_mpi ../../configs/ising_n3_validate.ini)
  # 2) validar:
  python3 cross_validate_t3a.py
"""
from __future__ import annotations
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common import qmpe, models  # noqa: E402

RESULTS = os.path.abspath(os.path.join(_HERE, "..", "results"))


def excitation_operator(N):
    """N_op = (1/N) sum_i |1><1|_i  (densidad de excitacion media)."""
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)   # |1><1|
    I2 = np.eye(2, dtype=complex)
    d = 2 ** N
    Nop = np.zeros((d, d), dtype=complex)
    for i in range(N):
        mats = [I2] * N
        mats[i] = P1
        op = mats[0]
        for k in range(1, N):
            op = np.kron(op, mats[k])
        Nop += op
    return Nop / N


def product_rho0(N, p0):
    """rho0 = |psi0><psi0| con |psi0> = prod_i (sqrt(1-p0)|0> + sqrt(p0)|1>)."""
    a, b = np.sqrt(1 - p0), np.sqrt(p0)
    psi1 = np.array([a, b], dtype=complex)
    psi = psi1
    for _ in range(N - 1):
        psi = np.kron(psi, psi1)
    return np.outer(psi, psi.conj())


def main():
    N = 3
    H, Ls, info = models.dissipative_ising(N=N, J=1.0, h=0.5, gamma=0.4, T=0.8)
    Nop = excitation_operator(N)

    max_err = 0.0
    print("=" * 60)
    print(f"  T3a cross-validation (C++ vs ecuacion maestra)  N={N}")
    print("=" * 60)
    for p0 in (0.1, 0.9):
        csvf = os.path.join(RESULTS, f"t3a_p0_{p0}.csv")
        if not os.path.exists(csvf):
            print(f"  FALTA {csvf}: genera primero los datos C++ (ver cabecera).")
            return
        data = np.loadtxt(csvf, delimiter=",", skiprows=1)
        t_cpp, n_cpp = data[:, 0], data[:, 1]

        # oraculo: evolucion densa de la ecuacion maestra
        rho0 = product_rho0(N, p0)
        times, rhos = qmpe.evolve_rk4(H, Ls, rho0, t_max=t_cpp[-1] + 1e-9,
                                      dt=5e-3, log_every=1)
        n_exact = np.array([np.real(np.trace(r @ Nop)) for r in rhos])
        # muestrear el exacto en los tiempos del C++
        n_exact_s = np.interp(t_cpp, times, n_exact)
        err = float(np.max(np.abs(n_cpp - n_exact_s)))
        max_err = max(max_err, err)
        print(f"  p0={p0}:  n_exc(0) C++={n_cpp[0]:.4f} exacto={n_exact_s[0]:.4f} | "
              f"n_exc(fin) C++={n_cpp[-1]:.4f} exacto={n_exact_s[-1]:.4f} | "
              f"max|dif|={err:.3e}")

    print("-" * 60)
    print(f"  max error C++ vs exacto = {max_err:.3e}")
    print("  OK (dentro del error estadistico M^-1/2)" if max_err < 0.02
          else "  REVISAR")


if __name__ == "__main__":
    main()
