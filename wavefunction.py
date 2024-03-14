import numpy as np
from matplotlib import pyplot as plt

import util


# lowest_energy, potential, effective_mass, length
def main(lowest_energy, length, position, potential, effective_mass):
    A, B, k = util.get_coefficients(lowest_energy, potential, effective_mass, length)

    zz = np.arange(0, position[-1] + 1e-12, 1e-12)
    Vz = np.empty_like(zz)
    kz = np.empty_like(zz)
    Az = np.empty_like(zz, dtype=complex)
    Bz = np.empty_like(zz, dtype=complex)
    PSI = np.empty_like(zz, dtype=complex)

    segment_indices = np.searchsorted(position, zz)
    segment_indices = np.clip(segment_indices, a_min=0, a_max=len(position) - 1)
    for i in range(len(zz)):
        segment_index = segment_indices[i]
        Vz[i] = potential[segment_index]
        kz[i] = np.real(k[segment_index])
        Az[i] = A[segment_index]
        Bz[i] = B[segment_index]
        PSI[i] = A[segment_index] * np.exp(1j * k[segment_index] * (zz[i] - position[segment_index])) + B[
            segment_index] * np.exp(
            -1j * k[segment_index] * (zz[i] - position[segment_index]))

    # Plot
    plt.figure(1)
    plt.plot(zz * 1e9, Vz, 'black', label='Potential - V(z)')  # Plotting potential
    plt.plot(zz * 1e9, PSI, 'purple', label='\u03A8')  # Plotting wavefunction
    plt.plot([0 * 1e9, position[3] * 1e9], [lowest_energy, lowest_energy], 'g--',
             label='E0')  # Minimum energy level line
    plt.title("\u03A8, InP/InGaAs/InP/InGaAs/InP")  # Title
    plt.xlabel("z (nm)")  # x-axis label
    plt.ylabel("Energy (eV)")  # y-axis label
    plt.legend()  # Show legend
    plt.show()
