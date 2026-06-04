#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mpdo_tebd.py  --  Tarea T3b: redes tensoriales para sistemas abiertos.

TEBD disipativo en la representacion VECTORIZADA del operador densidad (Zwolak &
Vidal, PRL 93, 207205 (2004); Weimer, Kshetrimayum, Orus, RMP 93, 015008 (2021),
arXiv:1907.07079). La matriz densidad rho se escribe como un "superket" |rho>>:
un MPS con dimension fisica 4 por sitio (el indice doblado fila/columna de cada
qubit). La ecuacion maestra d|rho>>/dt = L|rho>> se integra con TEBD: se
trotteriza e^{dt L} en compuertas locales de 2 sitios y se trunca el rango de
enlace a chi mediante SVD.

Convencion de indice fisico por sitio:  p = a*2 + b   (a = fila, b = columna).

Los superoperadores locales se construyen reutilizando `common.qmpe.build_liouvillian`
(que usa column-stacking vec) y permutando los indices al orden site-local.

Implementacion SERIAL; la paralela (bonds disjuntos por capa) esta en
tebd_parallel.py.
"""

from __future__ import annotations
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common import qmpe  # noqa: E402

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
SM = np.array([[0, 1], [0, 0]], dtype=complex)   # |0><1| baja
SP = np.array([[0, 0], [1, 0]], dtype=complex)   # |1><0| sube
I2 = np.eye(2, dtype=complex)


def n_bose(w, T):
    return 0.0 if w / T > 700 else 1.0 / (np.exp(w / T) - 1.0)


# =====================================================================
#  Permutaciones: column-stacking vec  ->  orden site-local p=a*2+b
# =====================================================================
def _perm_1site():
    """vec col-stack de rho 2x2: q = a + 2*b.  site-local: p = 2*a + b.
    Devuelve perm[p] = q  para reindexar el superoperador 4x4."""
    perm = np.zeros(4, dtype=int)
    for a in range(2):
        for b in range(2):
            p = a * 2 + b
            q = a + 2 * b
            perm[p] = q
    return perm


def _perm_2site():
    """vec col-stack de rho2 4x4 (2 qubits, kron: sitio0=MSB):
       R = a1*2+a2, C = b1*2+b2, q = R + 4*C.
       site-local: s = p1*4 + p2 = (a1*2+b1)*4 + (a2*2+b2).
    Devuelve perm[s] = q  (longitud 16)."""
    perm = np.zeros(16, dtype=int)
    for a1 in range(2):
        for a2 in range(2):
            for b1 in range(2):
                for b2 in range(2):
                    s = (a1 * 2 + b1) * 4 + (a2 * 2 + b2)
                    R = a1 * 2 + a2
                    C = b1 * 2 + b2
                    q = R + 4 * C
                    perm[s] = q
    return perm


_P1 = _perm_1site()
_P2 = _perm_2site()


def _to_local(S_colstack, perm):
    """Reindexa un superoperador de la base column-stacking a la site-local."""
    return S_colstack[np.ix_(perm, perm)]


def liouv_1site(H1, Ls1):
    """Liouvilliano local de 1 sitio (4x4) en base site-local."""
    S = qmpe.build_liouvillian(H1, Ls1)
    return _to_local(S, _P1)


def liouv_2site(H2, Ls2):
    """Liouvilliano local de 2 sitios (16x16) en base site-local.
    H2: hamiltoniano 4x4 (2 qubits); Ls2: operadores de salto 4x4."""
    S = qmpe.build_liouvillian(H2, Ls2)
    return _to_local(S, _P2)


# =====================================================================
#  Generadores de bond del modelo de Ising disipativo
# =====================================================================
def ising_bond_generators(N, J, h, gamma, T):
    """Construye, para cada bond b=(k,k+1), el generador local L_b (16x16) tal
    que sum_b L_b = L (Liouvilliano total). Reparto de terminos de 1 sitio:
    el sitio k se asigna al bond (k,k+1); el ultimo sitio (N-1) al bond (N-2,N-1).
    Devuelve lista de N-1 matrices 16x16.
    """
    nb = n_bose(2 * h if h > 0 else 1.0, T)
    Lm = np.sqrt(gamma * (nb + 1)) * SM
    Lp = np.sqrt(gamma * nb) * SP

    # superoperadores de 1 sitio embebidos en el espacio de 2 sitios
    L1 = liouv_1site(-h * SX, [Lm, Lp])           # 4x4
    I4 = np.eye(4, dtype=complex)
    L1_left = np.kron(L1, I4)                       # actua en el sitio izquierdo
    L1_right = np.kron(I4, L1)                      # actua en el sitio derecho

    # acoplamiento coherente -J Z⊗Z
    H2 = -J * np.kron(SZ, SZ)
    Lcoup = liouv_2site(H2, [])                     # 16x16

    bonds = []
    for k in range(N - 1):
        Lb = Lcoup + L1_left                        # sitio k siempre aqui
        if k + 1 == N - 1:                          # ultimo bond: anadir sitio N-1
            Lb = Lb + L1_right
        bonds.append(Lb)
    return bonds


def expm_gate(Lb, dt):
    """exp(dt * Lb) por descomposicion espectral (Lb 16x16, no hermitico)."""
    w, V = np.linalg.eig(Lb)
    Vinv = np.linalg.inv(V)
    return (V * np.exp(dt * w)) @ Vinv


# =====================================================================
#  MPS superket
# =====================================================================
class SuperketMPS:
    """MPS del superket |rho>> con tensores A[k] de forma (Dl, 4, Dr)."""

    def __init__(self, tensors):
        self.A = tensors
        self.N = len(tensors)

    @staticmethod
    def product(rho_sites):
        """Estado producto: rho = ⊗ rho_sites[k], cada rho_sites[k] una 2x2.
        El tensor de sitio es vec site-local p=a*2+b, forma (1,4,1)."""
        A = []
        for r in rho_sites:
            v = np.array([r[0, 0], r[0, 1], r[1, 0], r[1, 1]], dtype=complex)  # p=a*2+b
            A.append(v.reshape(1, 4, 1))
        return SuperketMPS(A)

    def apply_gate(self, k, G4, chi, tol=1e-12):
        """Aplica la compuerta de 2 sitios G (16x16 -> (4,4,4,4)) en los sitios
        (k,k+1) y trunca el enlace a chi por SVD."""
        Ak, Ak1 = self.A[k], self.A[k + 1]
        Dl = Ak.shape[0]; Dr = Ak1.shape[2]
        theta = np.tensordot(Ak, Ak1, axes=(2, 0))          # (Dl,4,4,Dr)
        G = G4.reshape(4, 4, 4, 4)                            # (p1',p2',p1,p2)
        theta = np.tensordot(G, theta, axes=([2, 3], [1, 2]))  # (p1',p2',Dl,Dr)
        theta = theta.transpose(2, 0, 1, 3)                  # (Dl,p1',p2',Dr)
        mat = theta.reshape(Dl * 4, 4 * Dr)
        U, S, Vh = np.linalg.svd(mat, full_matrices=False)
        # truncar
        keep = min(chi, np.sum(S > tol * S[0]) if S[0] > 0 else 1)
        keep = max(1, keep)
        U = U[:, :keep]; S = S[:keep]; Vh = Vh[:keep, :]
        self.A[k] = U.reshape(Dl, 4, keep)
        self.A[k + 1] = (np.diag(S) @ Vh).reshape(keep, 4, Dr)

    def trace(self):
        """Tr(rho) = <<I|rho>>, con <<I|_sitio = [1,0,0,1] (p=a*2+b)."""
        Ivec = np.array([1, 0, 0, 1], dtype=complex)
        v = np.ones((1,), dtype=complex)
        for k in range(self.N):
            v = np.tensordot(v, self.A[k], axes=(0, 0))      # (4,Dr)
            v = np.tensordot(Ivec, v, axes=(0, 0))           # (Dr,)
        return v[0]

    def expect_site(self, k, m_vec):
        """<<m_k|rho>> con covector m_vec (len 4) en el sitio k e identidad en
        el resto. Devuelve Tr(O_k rho) (sin normalizar por la traza)."""
        Ivec = np.array([1, 0, 0, 1], dtype=complex)
        v = np.ones((1,), dtype=complex)
        for s in range(self.N):
            cov = m_vec if s == k else Ivec
            v = np.tensordot(v, self.A[s], axes=(0, 0))
            v = np.tensordot(cov, v, axes=(0, 0))
        return v[0]

    def max_bond(self):
        return max(a.shape[2] for a in self.A[:-1]) if self.N > 1 else 1


# =====================================================================
#  Evolucion TEBD (Strang de 2o orden)
# =====================================================================
def excitation_density(mps):
    """n_exc = (1/N) sum_k Tr(|1><1|_k rho) / Tr(rho).
    covector de |1><1| (n=diag(0,1)): m_{(a,b)} = n_{b a} -> solo (1,1)=1, p=3."""
    m = np.array([0, 0, 0, 1], dtype=complex)
    tr = mps.trace()
    s = sum(mps.expect_site(k, m) for k in range(mps.N))
    return np.real(s / tr) / mps.N


def evolve_tebd(N, J, h, gamma, T, p0, t_max, dt, chi, log_every=20,
                observable=excitation_density):
    """Evoluciona el estado producto inicial (cada sitio sqrt(1-p0)|0>+sqrt(p0)|1>)
    con TEBD disipativo y registra el observable. Devuelve (times, valores, chi_max)."""
    bonds = ising_bond_generators(N, J, h, gamma, T)
    # compuertas de Strang: medio paso y paso completo
    G_half = [expm_gate(Lb, dt / 2) for Lb in bonds]
    G_full = [expm_gate(Lb, dt) for Lb in bonds]

    # estado inicial producto puro |psi><psi| por sitio
    a, b = np.sqrt(1 - p0), np.sqrt(p0)
    rho1 = np.outer([a, b], np.conj([a, b]))
    mps = SuperketMPS.product([rho1.copy() for _ in range(N)])

    even = list(range(0, N - 1, 2))
    odd = list(range(1, N - 1, 2))

    n_steps = int(np.ceil(t_max / dt))
    times, vals = [], []
    chi_max = 1
    for step in range(n_steps + 1):
        if step % log_every == 0:
            times.append(step * dt)
            vals.append(observable(mps))
        if step == n_steps:
            break
        # Strang: even(dt/2) odd(dt) even(dt/2)
        for k in even:
            mps.apply_gate(k, G_half[k], chi)
        for k in odd:
            mps.apply_gate(k, G_full[k], chi)
        for k in even:
            mps.apply_gate(k, G_half[k], chi)
        chi_max = max(chi_max, mps.max_bond())
    return np.array(times), np.array(vals), chi_max
