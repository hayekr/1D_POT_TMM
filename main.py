import numpy as np
import scipy.optimize as scipy

import util
import wavefunction

# Define Constants
q = 1.602176634e-19  # elementary charge in C
m0 = 9.10938356e-31  # electron rest mass in kg
hbar = 1.0545718e-34  # reduced Planck constant in J*s
Eg_InGaAs = 0.75  # Band gap of InGaAs in eV
Eg_InP = 1.344  # Band gap of InP in eV
m_e_InGaAs = 0.041 * m0  # electron effective mass in InGaAs
m_e_InP = 0.077 * m0  # electron effective mass in InP
# E_min = Eg_InP - (0.4 * (Eg_InP - Eg_InGaAs))
dE = 1e-3
conduction_discontinuity = 0.4 * (Eg_InP - Eg_InGaAs)  # 40% energy conduction band discontinuity


def main():
    print(
        'This tool will generate the lowest energy solution for a given potential profile. Two examples are included.')
    choice = input('Pick a Potential Profile (1 or 2) and press Enter to continue: ')

    if int(choice) == 1:
        L = np.array([20, 5, 20, 0]) * 1e-9
        #z = np.array([L[0], np.sum(L[0:2]), np.sum(L[0:3]), np.sum(L)])
        z = np.cumsum(L)
        V = np.array([Eg_InP, Eg_InP - 0.4 * (Eg_InP - Eg_InGaAs), Eg_InP, Eg_InP])
        m_star = np.array([m_e_InP, m_e_InGaAs, m_e_InP, m_e_InP], dtype=float)

    elif int(choice) == 2:
        L = np.array([20, 5, 1, 4, 0]) * 1e-9  # size of individual structures
        #z = np.array(
           # [L[0], np.sum(L[0:2]), np.sum(L[0:3]), np.sum(L[0:4]), np.sum(L)])  # position of each new structure
        np.cumsum(L)
        m_star = np.array([m_e_InP, m_e_InGaAs, m_e_InP, m_e_InGaAs, m_e_InP])  # effective mass of each
        V = np.array([Eg_InP, Eg_InP - 0.4 * (Eg_InP - Eg_InGaAs), Eg_InP, Eg_InP - 0.4 * (Eg_InP - Eg_InGaAs), Eg_InP])

    elif int(choice) == 3:
        L = np.array([10, 1, 5, 2, 10, 0]) * 1e-9
        #z = np.array([L[0], np.sum(L[0:2]), np.sum(L[0:3]), np.sum(L[0:4]), np.sum([L[0:5]]), np.sum(L)])
        z = np.cumsum(L)
        V = np.array([Eg_InP - 0.4 * (Eg_InP - Eg_InGaAs),
                      Eg_InP,
                      Eg_InP - 0.4 * (Eg_InP - Eg_InGaAs),
                      Eg_InP,
                      Eg_InP - 0.4 * (Eg_InP - Eg_InGaAs),
                      Eg_InP - 0.4 * (Eg_InP - Eg_InGaAs)])
        m_star = np.array([m_e_InGaAs, m_e_InP, m_e_InGaAs, m_e_InP, m_e_InGaAs, m_e_InGaAs])
    else:
        print('Invalid choice')
        exit(1)

    E_min = np.min(V + dE)

    # Get Lowest Energy Solution
    def f22val(E):
        return util.isolate_f22_TM(E, m_star, V, L)

    E1, = scipy.fsolve(f22val, E_min)
    print(f'Energy(eV) = {E1}')

    # Plot the wave function
    wavefunction.main(E1, L, z, V, m_star)


if __name__ == '__main__':
    main()
