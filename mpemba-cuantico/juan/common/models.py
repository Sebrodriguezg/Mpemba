#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
models.py  --  Modelos paradigmaticos resolubles (Seccion 6 del informe).

Cada constructor devuelve (H, Ls, info) donde H es el hamiltoniano, Ls la lista
de operadores de salto de Lindblad y `info` un dict con metadatos utiles
(omega0, T, p_eq analitico, etc.). Sirven de banco de pruebas para verificar el
nucleo numerico (qmpe.py) antes de escalar a HPC.

Modelos:
  tls                  -> qubit de amortiguamiento de amplitud generalizado (Sec 6.1)
  lambda_three_level   -> sistema Lambda de 3 niveles con DOS canales (muestra QMpE)
  dissipative_ising    -> cadena de Ising disipativa (Sec 6.3, caso que exige HPC)
"""

from __future__ import annotations
import numpy as np

# Matrices de Pauli y operadores de qubit
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
# Base (|0> = fundamental, indice 0 ; |1> = excitado, indice 1).
# Operador de BAJADA (emision) sigma_- = |0><1| lleva excitado -> fundamental.
# Operador de SUBIDA (absorcion) sigma_+ = |1><0| lleva fundamental -> excitado.
SIGMA_M = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1|  baja poblacion
SIGMA_P = np.array([[0, 0], [1, 0]], dtype=complex)   # |1><0|  sube poblacion
SP = SIGMA_P   # alias retrocompatible
SM = SIGMA_M
I2 = np.eye(2, dtype=complex)


def n_bose(omega: float, T: float) -> float:
    """Ocupacion de Bose-Einstein n_bar(T) = 1/(e^{omega/T} - 1)."""
    if omega / T > 700:
        return 0.0
    return 1.0 / (np.exp(omega / T) - 1.0)


# ---------------------------------------------------------------------
def tls(omega0=1.0, gamma=1.0, T=0.5):
    """Qubit de Davies (TLS), ecuacion (tls) del informe.

    dρ/dt = γ(n̄+1) D[σ-]ρ + γ n̄ D[σ+]ρ
    Relajacion monotona con tasa unica Γ = γ(2n̄+1); poblacion excitada de
    equilibrio p_eq = 1/(1+e^{ω0/T}). No muestra cruce: valida el integrador.
    """
    nb = n_bose(omega0, T)
    H = 0.5 * omega0 * SZ
    # Emision (tasa n̄+1) con sigma_- ; absorcion (tasa n̄) con sigma_+
    Ls = [np.sqrt(gamma * (nb + 1.0)) * SIGMA_M,
          np.sqrt(gamma * nb) * SIGMA_P]
    Gamma = gamma * (2 * nb + 1.0)
    p_eq = 1.0 / (1.0 + np.exp(omega0 / T))
    info = dict(name="TLS", omega0=omega0, gamma=gamma, T=T, n_bose=nb,
                Gamma=Gamma, p_eq=p_eq, d=2)
    return H, Ls, info


def tls_population_analytic(p0, t, Gamma, p_eq):
    """Solucion exacta p(t) = p_eq + (p0 - p_eq) e^{-Γ t} (para validacion)."""
    return p_eq + (p0 - p_eq) * np.exp(-Gamma * t)


# ---------------------------------------------------------------------
def lambda_three_level(omega=1.0, gamma1=1.5, gamma2=0.3, T=0.6):
    """Sistema Lambda de 3 niveles: dos estados base |0>,|1> y un excitado |2>.

    Dos canales de decaimiento del excitado hacia cada base, con tasas DISTINTAS
    gamma1 != gamma2 -> hay DOS modos de relajacion con tasas diferentes, que es
    el ingrediente minimo para el cruce de Mpemba (Sec 6.1: el TLS de un solo
    modo no basta). Energias: E0=0, E1=omega, E2=2*omega.
    """
    d = 3
    E = np.array([0.0, omega, 2 * omega])
    H = np.diag(E).astype(complex)

    def jump(i, j):  # |i><j|, transicion j -> i
        M = np.zeros((d, d), dtype=complex)
        M[i, j] = 1.0
        return M

    # ocupaciones termicas de cada transicion
    nb20 = n_bose(E[2] - E[0], T)
    nb21 = n_bose(E[2] - E[1], T)
    Ls = [
        np.sqrt(gamma1 * (nb20 + 1)) * jump(0, 2),  # 2 -> 0 (emision)
        np.sqrt(gamma1 * nb20) * jump(2, 0),        # 0 -> 2 (absorcion)
        np.sqrt(gamma2 * (nb21 + 1)) * jump(1, 2),  # 2 -> 1 (emision)
        np.sqrt(gamma2 * nb21) * jump(2, 1),        # 1 -> 2 (absorcion)
    ]
    info = dict(name="Lambda-3", omega=omega, gamma1=gamma1, gamma2=gamma2,
                T=T, d=d, energies=E)
    return H, Ls, info


# ---------------------------------------------------------------------
def dissipative_ising(N=3, J=1.0, h=0.5, gamma=0.4, T=0.8):
    """Cadena de Ising transversa con disipacion local (Sec 6.3).

    H = -J sum_i Z_i Z_{i+1} - h sum_i X_i ; canal de bajada local sqrt(gamma) σ-_i
    con su contraparte termica. d = 2^N, Liouvilliano 4^N x 4^N: este es el caso
    que EXIGE HPC. Para N<=4 corre en denso para validacion.
    """
    d = 2 ** N

    def op_at(op, i):
        """Inserta el operador 2x2 `op` en el sitio i de una cadena de N qubits."""
        mats = [I2] * N
        mats[i] = op
        out = mats[0]
        for k in range(1, N):
            out = np.kron(out, mats[k])
        return out

    H = np.zeros((d, d), dtype=complex)
    for i in range(N):
        H -= h * op_at(SX, i)
    for i in range(N - 1):
        H -= J * (op_at(SZ, i) @ op_at(SZ, i + 1))

    nb = n_bose(2 * h if h > 0 else 1.0, T)
    Ls = []
    for i in range(N):
        Ls.append(np.sqrt(gamma * (nb + 1)) * op_at(SM, i))
        Ls.append(np.sqrt(gamma * nb) * op_at(SP, i))
    info = dict(name=f"Ising-N{N}", N=N, J=J, h=h, gamma=gamma, T=T, d=d)
    return H, Ls, info


# ---------------------------------------------------------------------
def gibbs_state(H, T):
    """Estado termico rho = e^{-H/T}/Z (preparacion inicial a temperatura T)."""
    ev, U = np.linalg.eigh((H + H.conj().T) / 2)
    w = np.exp(-ev / T)
    w /= w.sum()
    return U @ np.diag(w) @ U.conj().T


def thermal_population_state(d, p_excited):
    """Estado diagonal con poblacion p_excited en el ultimo nivel (qubit/qudit)."""
    diag = np.zeros(d)
    diag[-1] = p_excited
    diag[0] = 1.0 - p_excited
    return np.diag(diag).astype(complex)
