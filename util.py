"""
This file contains utility functions for the 1D Schrödinger TMM for constant piecewise potential barrier.

Author: Robert Hayek
Date: 2024-02-29
Affiliation: Northwestern University
"""

from cmath import sqrt

import numpy as np
import scipy

# Global Constants
HBAR = 1.0545718e-34  # reduced Planck constant in J*s
Q = 1.602176634e-19  # elementary charge in C


def wave_vector(energy: float, potential: np.ndarray, effective_mass: np.ndarray) -> np.ndarray:
    """
    Calculate the wave vector for a given energy and potential profile.

    Parameters
    ----------
    energy : float
        The energy of the particle in eV.
    potential : np.ndarray
        A np.ndarray containing the potential profile of the system.
    effective_mass : np.ndarray
        A np.ndarray containing the effective mass profile of the system.

    Returns
    -------
    k : np.ndarray
        Array of wave vectors where each element corresponds to a potential region.
    """
    k = np.zeros(len(potential), dtype=complex)
    for i in range(len(k)):
        k[i] = (1 / HBAR) * sqrt((2 * effective_mass[i] * Q * (energy - potential[i])))
    return k


def p_value(k_n: float, k_n_plus_1: float, m_star_n: float, m_star_n_plus_1: float) -> float:
    """
    Calculate the p value for a given wave vector and effective mass.

    Parameters
    ----------
    k_n : complex
        The wave vector in the first region.
    k_n_plus_1 : complex
        The wave vector in the second region.
    m_star_n : float
        The effective mass in the first region.
    m_star_n_plus_1 : float
        The effective mass in the second region.

    Returns
    -------
    p : float
        The p value for the given region.
    """
    p = np.divide((m_star_n_plus_1 * k_n), (m_star_n * k_n_plus_1))
    return p


def transfer_matrix_F(k, length, p: float) -> np.ndarray:
    """
    Calculate the transfer matrix for a given region.

    Parameters
    ----------
    k : float
        The wave vector in the region.
    length : float
        The length of the region.
    p : float
        The p value for the region.

    Returns
    -------
    f : np.ndarray
        The transfer matrix for the given region.
    """
    p11 = (1 + p) * np.exp(1j * k * length)
    p12 = (1 - p) * np.exp(1j * k * length)
    p21 = (1 - p) * np.exp(-1j * k * length)
    p22 = (1 + p) * np.exp(-1j * k * length)
    f = 0.5 * np.array([[p11, p12],
                        [p21, p22]], dtype=complex)
    return f


def get_coefficients(energy: float, potential: np.ndarray, effective_mass: np.ndarray, length: np.ndarray):
    """
    Calculate the A and B coefficients for the wave function with a given lowest energy solution.
    :param energy: Lowest energy solution in eV
    :param potential: np.ndarray of potential profile
    :param effective_mass: np.ndarray of effective mass profile
    :param length: np.ndarray of length profile
    :return: A, B, k
    """
    k = wave_vector(energy, potential, effective_mass)

    AB = np.zeros((len(length), 2), dtype=complex)
    AB[0][0] = 0  # A0 = 0
    AB[0][1] = 1  # B0 = 1

    for i in range(len(potential) - 1):
        P = p_value(k[i].item(), k[i + 1].item(), effective_mass[i].item(), effective_mass[i + 1].item())
        F = transfer_matrix_F(k[i + 1], length[i + 1], P)  # 3x F for N+1 segments
        result = F @ AB[:][i]
        AB[i + 1][0] = result[0]
        AB[i + 1][1] = result[1]
        if i == (len(potential) - 1):
            AB[i + 1][1] = 0

    # A & B vectors
    A = AB[:, 0]
    B = AB[:, 1]
    return A, B, k


def isolate_f22_TM(E, m, V, L):
    k = wave_vector(E, V, m)
    F = np.eye(2)
    for j in range(len(k) - 1):
        P = p_value(k[j].item(), k[j + 1].item(), m[j], m[j + 1])
        F = transfer_matrix_F(k[j + 1], L[j + 1], P) @ F
    return np.real(F[1, 1])


def transfer_matrix_b(k, length, p: float):
    """
    Calculate the transfer matrix for a given region.

    Parameters
    ----------
    k : float
        The wave vector in the region.
    length : float
        The length of the region.
    p : float
        The p value for the region.

    Returns
    -------
    b : np.ndarray
        The transfer matrix for the given region.
    """
    return np.linalg.inv(transfer_matrix_F(k, length, p))


def transmission(energy, mass, potential, length):
    k = wave_vector(energy, potential, mass)
    B = np.eye(2)
    for i in range(len(k) - 1):
        p = p_value(k[i], k[i + 1], mass[i], mass[i + 1])
        B = B @ transfer_matrix_b(p, k[i + 1], length[i + 1])
    b11 = abs(B[0, 0])
    return np.divide(1, b11)