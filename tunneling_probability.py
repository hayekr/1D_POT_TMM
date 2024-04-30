import numpy as np
import util
import matplotlib.pyplot as plt


def b_matrix(p_value, wave_vector, length):
    B = np.linalg.inv(util.transfer_matrix_F(wave_vector, length, p_value))
    return B


def tunneling_probability(energy, mass, potential, length):
    wave_vector = util.wave_vector(energy, potential, mass)
    B = np.eye(2)
    for i in range(len(wave_vector) - 1):
        p = util.p_value(wave_vector[i], wave_vector[i + 1], mass[i], mass[i + 1])
        B = B @ b_matrix(p, wave_vector[i+1], length[i+1])
    b11 = abs(B[0, 0])
    return np.divide(1, b11)

def accurate_tunneling_probability(energy, mass, potential, length):
    mass[:] = [m0 * (energy + 0.984)/24 if x == m_e_InGaAs else m0 * (energy + 1.694)/22 for x in mass]
    wave_vector = util.wave_vector(energy, potential, mass)
    B = np.eye(2)
    for i in range(len(wave_vector) - 1):
        p = util.p_value(wave_vector[i], wave_vector[i + 1], mass[i], mass[i + 1])
        B = B @ b_matrix(p, wave_vector[i + 1], length[i + 1])
    b11 = abs(B[0, 0])
    return np.divide(1, b11)

# Constants
m0 = 9.10938356e-31  # electron mass (kg)
hbar = 1.0545718e-34  # reduced Planck constant (J s)
q = 1.60217662e-19  # elementary charge (C)

m_e_InGaAs = 0.041 * m0
m_e_InP = 0.007 * m0

E_g_InGaAs = 0.75  # band gap (J)
E_g_InP = 1.344  # band gap (J)

C_band_disc = 0.4 * (E_g_InP - E_g_InGaAs)  # Conduct band discontinuity (J)

E_G_InP_new = E_g_InGaAs + C_band_disc  # New band gap (J)

L = np.array([10, 1, 5, 2, 10]) * 1e-9  # barrier thickness (m)
z = np.array([L[0], np.sum(L[0:2]), np.sum(L[0:3]), np.sum(L[0:4]), np.sum(L)])
m = np.array([m_e_InGaAs, m_e_InP, m_e_InGaAs, m_e_InP, m_e_InGaAs])
V = np.array([E_g_InGaAs, E_G_InP_new, E_g_InGaAs, E_G_InP_new, E_g_InGaAs])

energy_range = np.linspace(0, 2, 10000)
tunneling_probabilities = np.array([tunneling_probability(energy_range[i], m, V, L) for i in range(len(energy_range))])

acc_tunneling = np.array(np.array([accurate_tunneling_probability(energy_range[i], m, V, L) for i in range(len(energy_range))]))

plt.plot(tunneling_probabilities, energy_range, 'b', label='Effective Mass')
plt.plot(acc_tunneling, energy_range, 'r', label='Energy Dependent mass')
plt.xlabel('Tunneling Probability')
plt.ylabel('Energy (eV)')
plt.title('Tunneling Probability vs Energy')
plt.grid()
plt.legend()
plt.show()