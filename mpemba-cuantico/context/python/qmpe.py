#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qmpe.py  --  Implementacion de referencia del efecto Mpemba cuantico (QMpE).

Acompana al informe `mpemba_cuantico.tex`. Cada funcion implementa de forma
explicita y verificable los objetos definidos en el documento:

    Seccion 2  -> ecuacion maestra GKSL                 build_liouvillian, apply_liouvillian
    Seccion 3  -> descomposicion espectral              spectrum, overlaps
    Seccion 4  -> distancias y criterio de cruce        distances, crossing_time
    Seccion 6  -> modelos resolubles                    models.py
    Seccion 7  -> tareas T1 (modos lentos), T2 (evol.)  slow_modes, evolve_*

Convencion: hbar = kB = 1. Vectorizacion por apilado de columnas
(`column-stacking`): vec(rho) = rho.reshape(-1, order='F').

Solo depende de numpy (sin scipy), de modo que es directamente ejecutable en el
entorno del proyecto. Esta pensada para ESTUDIAR el fenomeno en pocos cuerpos y
como especificacion de referencia de la version HPC en C++ (carpeta ../cpp).
"""

from __future__ import annotations
import numpy as np


# =====================================================================
#  Vectorizacion (Seccion 7.1 del informe)
# =====================================================================
def vec(rho: np.ndarray) -> np.ndarray:
    """vec(rho) por apilado de columnas (column-stacking)."""
    return rho.reshape(-1, order="F")


def unvec(v: np.ndarray, d: int) -> np.ndarray:
    """Inversa de vec: reconstruye la matriz d x d."""
    return v.reshape((d, d), order="F")


def build_liouvillian(H: np.ndarray, Ls: list[np.ndarray]) -> np.ndarray:
    """
    Superoperador de Lindblad como matriz densa L (d^2 x d^2), ecuacion (vecL):

        L = -i (I⊗H - H^T⊗I)
            + sum_mu [ L_mu*⊗L_mu - 1/2 ( I⊗(L_mu† L_mu) + (L_mu^T L_mu*)⊗I ) ]

    Usa la identidad vec(A X B) = (B^T ⊗ A) vec(X). El espectro de L coincide
    con el del superoperador Lop del informe.
    """
    d = H.shape[0]
    I = np.eye(d, dtype=complex)
    L = -1j * (np.kron(I, H) - np.kron(H.T, I))
    for Lk in Ls:
        Lk = Lk.astype(complex)
        LdL = Lk.conj().T @ Lk
        L += np.kron(Lk.conj(), Lk)
        L -= 0.5 * (np.kron(I, LdL) + np.kron(LdL.T, I))
    return L


def apply_liouvillian(H: np.ndarray, Ls: list[np.ndarray], V: np.ndarray) -> np.ndarray:
    """
    Accion 'matrix-free' del Liouvilliano sobre un operador V (Seccion 7.2):

        L[V] = -i[H, V] + sum_mu ( L_mu V L_mu† - 1/2 {L_mu† L_mu, V} )

    No materializa la matriz d^2 x d^2: este es el nucleo que escala a HPC.
    """
    out = -1j * (H @ V - V @ H)
    for Lk in Ls:
        Ld = Lk.conj().T
        LdL = Ld @ Lk
        out += Lk @ V @ Ld - 0.5 * (LdL @ V + V @ LdL)
    return out


# =====================================================================
#  Descomposicion espectral  (Seccion 3, tarea T1)
# =====================================================================
class Spectrum:
    """Espectro del Liouvilliano: autovalores y autooperadores derecho/izquierdo.

    Atributos
    ---------
    eigvals : (d^2,)        autovalores lambda_k ordenados por |Re| creciente
    r_ops   : lista de (d,d) autooperadores derechos  r_k  (L[r_k] = lam_k r_k)
    l_ops   : lista de (d,d) autooperadores izquierdos l_k (biortonormales)
    rho_ss  : (d,d)         estado estacionario r_1 (lambda_1 = 0), traza 1
    """

    def __init__(self, H, Ls):
        self.H = H
        self.Ls = Ls
        self.d = H.shape[0]
        self._diagonalize()

    def _diagonalize(self):
        d = self.d
        L = build_liouvillian(self.H, self.Ls)
        # Autovectores derechos
        lam, R = np.linalg.eig(L)
        # Autovectores izquierdos via inversa: filas de R^{-1} son biortonormales
        Rinv = np.linalg.inv(R)
        # Orden por parte real creciente en modulo (modo estacionario primero)
        order = np.argsort(np.abs(lam.real))
        lam = lam[order]
        R = R[:, order]
        Rinv = Rinv[order, :]
        self.eigvals = lam
        self.r_ops = [unvec(R[:, k], d) for k in range(d * d)]
        self.l_ops = [unvec(Rinv[k, :].conj(), d) for k in range(d * d)]
        # Estado estacionario: lambda_1 ~ 0, normalizado a traza 1
        rss = self.r_ops[0]
        self.rho_ss = rss / np.trace(rss)

    def overlaps(self, rho0: np.ndarray) -> np.ndarray:
        """Coeficientes a_k = Tr(l_k† rho0)  (solapamiento inicial, ec. spectral).

        a_2 = 0 es la condicion de Mpemba fuerte (ec. strong del informe).
        """
        return np.array([np.trace(lk.conj().T @ rho0) for lk in self.l_ops])

    def relaxation_time(self) -> float:
        """tau = 1 / |Re(lambda_2)|: escala de relajacion del modo lento."""
        return 1.0 / abs(self.eigvals[1].real)


def slow_modes(H, Ls, k=4):
    """Tarea T1: devuelve los k autovalores mas lentos y sus autooperadores.

    En la version HPC esto se hace con Arnoldi shift-invert (SLEPc); aqui, para
    pocos cuerpos, se obtiene del espectro denso completo.
    """
    sp = Spectrum(H, Ls)
    return sp.eigvals[:k], sp.r_ops[:k], sp.l_ops[:k], sp.rho_ss


# =====================================================================
#  Evolucion temporal  (tarea T2)
# =====================================================================
def evolve_spectral(sp: Spectrum, rho0: np.ndarray, times: np.ndarray):
    """Evolucion EXACTA por suma de modos (ec. spectral del informe):

        rho_t = rho_ss + sum_{k>=2} e^{lam_k t} a_k r_k

    Devuelve un array (len(times), d, d).
    """
    a = sp.overlaps(rho0)
    d = sp.d
    out = np.empty((len(times), d, d), dtype=complex)
    for n, t in enumerate(times):
        rho = np.zeros((d, d), dtype=complex)
        for k in range(d * d):
            rho += np.exp(sp.eigvals[k] * t) * a[k] * sp.r_ops[k]
        out[n] = rho
    return out


def evolve_rk4(H, Ls, rho0: np.ndarray, t_max: float, dt: float, log_every: int = 1):
    """Evolucion por RK4 'matrix-free' (la via que escala en HPC, tarea T2).

    Integra d rho/dt = L[rho] aplicando apply_liouvillian (sin formar L).
    Devuelve (times, rhos).
    """
    n_steps = int(np.ceil(t_max / dt))
    rho = rho0.astype(complex).copy()
    times, rhos = [], []
    for step in range(n_steps + 1):
        if step % log_every == 0:
            times.append(step * dt)
            rhos.append(rho.copy())
        if step < n_steps:
            k1 = apply_liouvillian(H, Ls, rho)
            k2 = apply_liouvillian(H, Ls, rho + 0.5 * dt * k1)
            k3 = apply_liouvillian(H, Ls, rho + 0.5 * dt * k2)
            k4 = apply_liouvillian(H, Ls, rho + dt * k3)
            rho = rho + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return np.array(times), np.array(rhos)


# =====================================================================
#  Distancias al equilibrio  (Seccion 4.1)
# =====================================================================
def trace_distance(rho, rss):
    """D_tr = 1/2 ||rho - rss||_1  (suma de |autovalores| de la diferencia)."""
    diff = rho - rss
    ev = np.linalg.eigvalsh((diff + diff.conj().T) / 2)  # diff es hermitica
    return 0.5 * np.sum(np.abs(ev))


def hs_distance(rho, rss):
    """D_HS = ||rho - rss||_2 = sqrt(Tr[(rho-rss)^2])."""
    diff = rho - rss
    return np.sqrt(np.real(np.trace(diff @ diff)))


def kl_divergence(rho, rss):
    """KL entre las poblaciones (diagonales en la base de energia)."""
    p = np.real(np.diag(rho))
    q = np.real(np.diag(rss))
    mask = p > 1e-300
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def relative_entropy(rho, rss):
    """S(rho||rss) = Tr[rho(ln rho - ln rss)]  (entropia relativa cuantica)."""
    er, Ur = np.linalg.eigh((rho + rho.conj().T) / 2)
    es, Us = np.linalg.eigh((rss + rss.conj().T) / 2)
    er = np.clip(er, 1e-300, None)
    es = np.clip(es, 1e-300, None)
    log_rho = Ur @ np.diag(np.log(er)) @ Ur.conj().T
    log_rss = Us @ np.diag(np.log(es)) @ Us.conj().T
    return float(np.real(np.trace(rho @ (log_rho - log_rss))))


# =====================================================================
#  Informacion de Fisher cuantica  (Seccion 5.6, ec. qfi)
# =====================================================================
def qfi_temperature(populations, dT_populations):
    """F_T = sum_i (d_T p_i)^2 / p_i  para estados diagonales en energia."""
    p = np.clip(populations, 1e-15, None)
    return float(np.sum(dT_populations ** 2 / p))


# =====================================================================
#  Deteccion del cruce de Mpemba  (Seccion 4.2)
# =====================================================================
def crossing_time(times, D_hot, D_cold):
    """Primer t* tras el cual D_hot < D_cold de forma persistente.

    Devuelve t* o None si no hay cruce. Firma del efecto Mpemba: la preparacion
    'caliente' (parte mas lejos) termina mas cerca del equilibrio.
    """
    times = np.asarray(times)
    diff = np.asarray(D_hot) - np.asarray(D_cold)
    started_above = diff[0] > 0
    if not started_above:
        return None
    for n in range(1, len(times)):
        if diff[n] < 0 and np.all(diff[n:] < 1e-12):
            # interpolacion lineal del cruce
            t0, t1 = times[n - 1], times[n]
            d0, d1 = diff[n - 1], diff[n]
            return float(t0 - d0 * (t1 - t0) / (d1 - d0))
    return None
