#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arnoldi_lindblad.py  --  Tarea T1: modos lentos del Liouvilliano via Arnoldi-Lindblad.

Implementacion SERIAL de referencia (numpy puro) del algoritmo de
Minganti & Huybrechts, "Arnoldi-Lindblad time evolution: faster-than-the-clock
algorithm for the spectrum of time-independent and Floquet open quantum systems",
Quantum 6, 649 (2022)  [arXiv:2109.13883].

----------------------------------------------------------------------------
Idea central
----------------------------------------------------------------------------
T1 busca los pocos autovalores del Liouvilliano L mas cercanos a 0 (el estado
estacionario lambda_1 = 0 y los modos lentos lambda_2, lambda_3, ...) junto con
sus autooperadores derechos r_k e izquierdos l_k, para diagnosticar el efecto
Mpemba y calcular los solapamientos a_k = Tr(l_k^dag rho_0).

En vez de diagonalizar L (denso, O(d^6)) o de usar shift-invert (que exige
resolver sistemas lineales dispersos), Arnoldi-Lindblad construye el subespacio
de Krylov con el PROPAGADOR

        P(tau) = e^{tau L},

cuya accion P[V] = e^{tau L}[V] se calcula SIN materializar L, integrando la
ecuacion maestra  dV/dt = L[V]  en [0, tau] con RK4 matrix-free (el mismo kernel
`apply_liouvillian` de la tarea T2). El espectro de P y el de L estan ligados por

        mu_k = e^{tau lambda_k},      lambda_k = ln(mu_k) / tau.

Como Re(lambda_k) <= 0, todos los mu_k caen en el disco unidad |mu| <= 1; el
estado estacionario es mu_1 = 1 y los MODOS LENTOS son los de mayor modulo |mu_k|
(los mas cercanos a la frontera). La iteracion de Arnoldi converge primero a los
autovalores DOMINANTES en modulo de P -> exactamente los modos lentos de L. De
ahi el "faster than the clock": basta un subespacio de Krylov pequeno (unas pocas
acciones del propagador) para extraer el estacionario y el gap, en lugar de
evolucionar el sistema durante el tiempo fisico tau_relax = 1/|Re(lambda_2)|.

----------------------------------------------------------------------------
Coste y paralelizacion (ver version C++ en ../cpp)
----------------------------------------------------------------------------
La primitiva dominante es la accion matrix-free de L (SpMV), llamada
n_steps_por_tau * (1 + m) * restarts veces. Arnoldi es SECUENCIALMENTE acoplado
(cada vector de Krylov depende del anterior), de modo que el paralelismo
escalable vive DENTRO del SpMV: OpenMP intranodo + distribucion MPI del operador
d x d. Block-Arnoldi (varios vectores semilla) anade un eje paralelo extra.
"""

from __future__ import annotations
import sys
import os
import numpy as np

# Permitir importar el oraculo `common` (qmpe, models) tanto si se ejecuta desde
# T1/python como desde la raiz del entregable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common import qmpe  # noqa: E402


# =====================================================================
#  Producto interno de Hilbert-Schmidt sobre el espacio de operadores
#     <A, B>_HS = Tr(A^dag B)
#  Trabajamos en el espacio de operadores d x d (no vectorizado): es la
#  formulacion natural matrix-free.
# =====================================================================
def hs_inner(A: np.ndarray, B: np.ndarray) -> complex:
    """Producto interno de Hilbert-Schmidt <A,B> = Tr(A^dag B) = sum conj(A)*B."""
    return np.vdot(A, B)  # vdot conjuga el primer argumento; suma elemento a elemento


def hs_norm(A: np.ndarray) -> float:
    """Norma de Hilbert-Schmidt ||A|| = sqrt(Tr(A^dag A))."""
    return float(np.sqrt(np.real(np.vdot(A, A))))


# =====================================================================
#  Mapas duales del Liouvilliano (para los autooperadores izquierdos)
#     L[O]   = -i[H,O] + sum_mu ( L_mu O L_mu^dag - 1/2 {L_mu^dag L_mu, O} )
#     L^dag[O] = +i[H,O] + sum_mu ( L_mu^dag O L_mu - 1/2 {L_mu^dag L_mu, O} )
#  L^dag es el adjunto respecto a <.,.>_HS y genera la dinamica de Heisenberg.
# =====================================================================
def apply_liouvillian_dual(H, Ls, O):
    """Accion matrix-free del Liouvilliano DUAL L^dag (imagen de Heisenberg)."""
    out = 1j * (H @ O - O @ H)
    for Lk in Ls:
        Ld = Lk.conj().T
        LdL = Ld @ Lk
        out += Ld @ O @ Lk - 0.5 * (LdL @ O + O @ LdL)
    return out


# =====================================================================
#  Accion del propagador  P(tau)[V] = e^{tau L}[V]  via RK4 matrix-free
# =====================================================================
def propagator_action(H, Ls, V, tau, dt, dual=False):
    """Aplica e^{tau L}[V] (o e^{tau L^dag}[V] si dual=True) integrando con RK4.

    Devuelve (resultado, n_spmv): el numero de aplicaciones del Liouvilliano
    realizadas (4 por paso RK4), para contabilizar el coste real.
    """
    apply = apply_liouvillian_dual if dual else qmpe.apply_liouvillian
    n_steps = max(1, int(np.ceil(tau / dt)))
    h = tau / n_steps
    X = V.astype(complex).copy()
    n_spmv = 0
    for _ in range(n_steps):
        k1 = apply(H, Ls, X)
        k2 = apply(H, Ls, X + 0.5 * h * k1)
        k3 = apply(H, Ls, X + 0.5 * h * k2)
        k4 = apply(H, Ls, X + h * k3)
        X = X + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        n_spmv += 4
    return X, n_spmv


# =====================================================================
#  Iteracion de Arnoldi sobre un operador lineal generico `matvec`
#  en el espacio de operadores con producto interno de Hilbert-Schmidt.
# =====================================================================
def arnoldi(matvec, V0, m, tol=1e-12):
    """Construye una base ortonormal de Krylov K_m(matvec, V0).

    Parametros
    ----------
    matvec : callable (d,d)->(d,d)   accion del operador lineal (aqui, P(tau))
    V0     : (d,d)                   operador semilla
    m      : int                     dimension maxima del subespacio
    tol    : float                   tolerancia de breakdown (invariancia)

    Devuelve
    --------
    Q : lista de (d,d)   base ortonormal {q_0,...,q_{k}}  (HS-ortonormal)
    Hess : (k, k)        matriz de Hessenberg superior  (proyeccion de matvec)
    n_spmv : int         numero de aplicaciones de L acumuladas
    """
    beta = hs_norm(V0)
    if beta < tol:
        raise ValueError("vector semilla nulo")
    Q = [V0 / beta]
    Hbig = np.zeros((m + 1, m), dtype=complex)
    total_spmv = 0
    last = m
    for j in range(m):
        w, n_spmv = matvec(Q[j])
        total_spmv += n_spmv
        # Gram-Schmidt modificado (estable)
        for i in range(j + 1):
            Hbig[i, j] = hs_inner(Q[i], w)
            w = w - Hbig[i, j] * Q[i]
        hjp = hs_norm(w)
        Hbig[j + 1, j] = hjp
        if hjp < tol:                 # subespacio invariante alcanzado
            last = j + 1
            break
        Q.append(w / hjp)
    k = min(last, len(Q))
    return Q[:k], Hbig[:k, :k], total_spmv


# =====================================================================
#  Extraccion de modos lentos a partir de un lado (derecho o izquierdo)
# =====================================================================
def _slow_modes_one_side(H, Ls, tau, dt, m, restarts, k, seed, dual):
    """Arnoldi(+restart explicito) sobre el propagador P(tau) de un solo lado.

    Devuelve (lambdas, ops, n_spmv): los k autovalores mas lentos de L y sus
    autooperadores (derechos si dual=False, izquierdos si dual=True), junto al
    coste en SpMV.
    """
    d = H.shape[0]
    rng = np.random.default_rng(seed)

    def matvec(V):
        return propagator_action(H, Ls, V, tau, dt, dual=dual)

    # Semilla: operador hermitico aleatorio (tiene solapamiento generico con todos
    # los modos fisicos; el estacionario es hermitico).
    V0 = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    V0 = V0 + V0.conj().T

    total_spmv = 0
    ritz_lam = None
    ritz_ops = None
    for _ in range(max(1, restarts)):
        Q, Hess, n_spmv = arnoldi(matvec, V0, m)
        total_spmv += n_spmv
        mu, Y = np.linalg.eig(Hess)            # autovalores de P proyectado
        # mu_k = e^{tau lambda_k}; modos lentos = mayor modulo
        order = np.argsort(-np.abs(mu))
        mu = mu[order]
        Y = Y[:, order]
        lam = np.log(mu) / tau                  # lambda_k = ln(mu_k)/tau
        # Autooperadores de Ritz: combinacion de la base de Krylov
        ops = []
        for c in range(Y.shape[1]):
            R = np.zeros((d, d), dtype=complex)
            for i, q in enumerate(Q):
                R += Y[i, c] * q
            ops.append(R)
        ritz_lam, ritz_ops = lam, ops
        # Restart explicito: re-sembrar con la suma de los k modos dominantes
        # (subespacio invariante aproximado), acelera la convergencia.
        if len(ops) >= 1:
            V0 = np.zeros((d, d), dtype=complex)
            for c in range(min(k, len(ops))):
                V0 = V0 + ops[c]
            V0 = V0 + 1e-3 * (rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
            V0 = V0 + V0.conj().T
    kk = min(k, len(ritz_lam))
    return ritz_lam[:kk], ritz_ops[:kk], total_spmv


# =====================================================================
#  API publica de la tarea T1
# =====================================================================
class ArnoldiLindbladResult:
    """Resultado de T1: modos lentos del Liouvilliano y diagnostico de Mpemba."""

    def __init__(self, eigvals, r_ops, l_ops, rho_ss, n_spmv):
        self.eigvals = eigvals      # (k,)  lambda_1=0, lambda_2, ... ordenados por |Re|
        self.r_ops = r_ops          # autooperadores derechos r_k
        self.l_ops = l_ops          # autooperadores izquierdos l_k (biortonormales)
        self.rho_ss = rho_ss        # estado estacionario r_1 normalizado a traza 1
        self.n_spmv = n_spmv        # coste total en aplicaciones de L

    def overlaps(self, rho0):
        """a_k = Tr(l_k^dag rho0). a_2 = 0  <=>  Mpemba fuerte (ec. strong)."""
        return np.array([np.trace(lk.conj().T @ rho0) for lk in self.l_ops])

    def gap(self):
        """Gap de Liouvilliano |Re(lambda_2)| (inverso del tiempo de relajacion)."""
        return abs(self.eigvals[1].real) if len(self.eigvals) > 1 else 0.0


def slow_modes(H, Ls, k=4, tau=None, dt=2e-3, m=40, restarts=3, seed=12345):
    """Tarea T1 (Arnoldi-Lindblad): los k modos mas lentos de L y sus operadores.

    Parametros
    ----------
    H, Ls   : modelo (hamiltoniano y operadores de salto)
    k       : numero de modos lentos a extraer (incluye el estacionario)
    tau     : paso del propagador e^{tau L}. Si None, se elige automaticamente a
              partir de una estimacion de la escala de relajacion.
    dt      : paso de RK4 para la accion del propagador
    m       : dimension del subespacio de Krylov
    restarts: numero de reinicios explicitos
    seed    : semilla del operador inicial

    Devuelve un ArnoldiLindbladResult con autovalores, r_k, l_k (biortonormales),
    estado estacionario y coste en SpMV.
    """
    d = H.shape[0]
    if tau is None:
        # Heuristica: tau ~ escala que separa bien los modos lentos. Usamos una
        # fraccion de una estimacion grosera del tiempo de relajacion via la norma
        # de L sobre un operador de prueba.
        probe = np.eye(d, dtype=complex) / d
        Lprobe = qmpe.apply_liouvillian(H, Ls, probe + 0.1 * np.diag(np.arange(d)))
        scale = hs_norm(Lprobe) + 1e-12
        tau = max(1e-2, 0.5 / scale)

    lam_r, r_ops, spmv_r = _slow_modes_one_side(
        H, Ls, tau, dt, m, restarts, k, seed, dual=False)
    lam_l, l_ops, spmv_l = _slow_modes_one_side(
        H, Ls, tau, dt, m, restarts, k, seed + 777, dual=True)

    # Ordenar ambos lados por |Re(lambda)| creciente (estacionario primero) y
    # emparejar derecho/izquierdo por cercania de autovalor.
    order_r = np.argsort(np.abs(lam_r.real))
    lam_r = lam_r[order_r]
    r_ops = [r_ops[i] for i in order_r]
    paired_l = []
    used = set()
    for lr in lam_r:
        # l_k corresponde a lambda_k (no su conjugado): emparejar por |lam_l - lr|
        j_best, d_best = 0, np.inf
        for j, ll in enumerate(lam_l):
            if j in used:
                continue
            dd = abs(ll - lr)
            if dd < d_best:
                d_best, j_best = dd, j
        used.add(j_best)
        paired_l.append(l_ops[j_best])
    l_ops = paired_l

    # Biortonormalizacion: imponer Tr(l_j^dag r_k) = delta_jk resolviendo el
    # pequeno sistema de solapamientos M_{jk} = Tr(l_j^dag r_k) -> l <- M^{-dag} l.
    K = len(r_ops)
    M = np.array([[np.trace(l_ops[j].conj().T @ r_ops[i]) for i in range(K)]
                  for j in range(K)], dtype=complex)
    Minv = np.linalg.inv(M)
    new_l = []
    for i in range(K):
        Li = np.zeros_like(l_ops[0])
        for j in range(K):
            Li = Li + np.conj(Minv[i, j]) * l_ops[j]
        new_l.append(Li)
    l_ops = new_l

    # Estado estacionario: r_1 normalizado a traza 1 (debe ser ~hermitico, traza>0)
    rss = r_ops[0]
    tr = np.trace(rss)
    rho_ss = rss / tr if abs(tr) > 1e-12 else rss
    rho_ss = (rho_ss + rho_ss.conj().T) / 2  # simetrizar ruido numerico

    return ArnoldiLindbladResult(lam_r, r_ops, l_ops, rho_ss, spmv_r + spmv_l)


# =====================================================================
#  "Faster than the clock": coste de Arnoldi-Lindblad vs evolucion directa
# =====================================================================
def steady_state_cost_comparison(H, Ls, dt=2e-3, tol=1e-8):
    """Compara el coste (en SpMV) de alcanzar el estado estacionario por:
      (a) evolucion temporal directa hasta convergencia (RK4),
      (b) Arnoldi-Lindblad.
    Devuelve dict con n_spmv de cada metodo y el estado estacionario de cada uno.
    """
    d = H.shape[0]
    # Referencia densa del estacionario
    sp = qmpe.Spectrum(H, Ls)
    rho_ss_ref = sp.rho_ss

    # (a) Evolucion directa desde el maximamente mixto
    rho = np.eye(d, dtype=complex) / d
    n_spmv_evol = 0
    t = 0.0
    max_t = 200.0
    while t < max_t:
        k1 = qmpe.apply_liouvillian(H, Ls, rho)
        k2 = qmpe.apply_liouvillian(H, Ls, rho + 0.5 * dt * k1)
        k3 = qmpe.apply_liouvillian(H, Ls, rho + 0.5 * dt * k2)
        k4 = qmpe.apply_liouvillian(H, Ls, rho + dt * k3)
        rho = rho + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        n_spmv_evol += 4
        t += dt
        if qmpe.hs_distance(rho, rho_ss_ref) < tol:
            break

    # (b) Arnoldi-Lindblad
    res = slow_modes(H, Ls, k=4, dt=dt, m=40, restarts=3)
    err_arnoldi = qmpe.hs_distance(res.rho_ss, rho_ss_ref)

    return {
        "n_spmv_evolution": n_spmv_evol,
        "n_spmv_arnoldi": res.n_spmv,
        "err_evolution": qmpe.hs_distance(rho, rho_ss_ref),
        "err_arnoldi": err_arnoldi,
        "speedup": n_spmv_evol / max(1, res.n_spmv),
    }
