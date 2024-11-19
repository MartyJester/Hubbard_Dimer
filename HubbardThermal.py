import numpy as np
import matplotlib.pyplot as plt


def h_1p(t, V):
    matrix = np.array([
        [-(V / 2), 0, -t, 0],
        [0, -(V / 2), 0, -t],
        [-t, 0, V / 2, 0],
        [0, -t, 0, V / 2]])
    return matrix


def h_2p(t, V, U):
    matrix = np.array([
        [U - V, -t, t, 0, 0, 0],
        [-t, 0, 0, -t, 0, 0],
        [t, 0, 0, t, 0, 0],
        [0, -t, t, U + V, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    return matrix


def en_0p(t, V, U):
    return np.array([0])


def en_1p(t, V, U):
    return np.linalg.eigvals(h_1p(t, V))


def en_2p(t, V, U):
    return np.linalg.eigvals(h_2p(t, V, U))


def en_3p(t, V, U):
    return en_1p(t, V, U) + U


def en_4p(t, V, U):
    return np.array([2 * U])


def energy_combined(t, V, U):
    return [en_0p(t, V, U), en_1p(t, V, U), en_2p(t, V, U), en_3p(t, V, U), en_4p(t, V, U)]


def can_z_part_func(t, V, U, tau, n):
    if n >= 0 and n <= 3:
        return np.sum(np.exp(-1 / tau * energy_combined(t, V, U)[n]))
    else:
        print('error in can_z_part_func')


def energies_concat(t, V, U, mu):
    array_mu = np.array([0, 1, 2, 3, 4]) * mu
    energies = energy_combined(t, V, U)
    result_rows = []
    for i, row in enumerate(energies):
        subtracted_row = row - array_mu[i]
        result_rows.append(subtracted_row)
    energies = np.concatenate(result_rows)
    return energies


def granc_z_part_func(t, V, U, tau, mu):
    energies = energies_concat(t, V, U, mu)
    zpart = np.sum(np.exp(energies * (-1 / tau)))
    return zpart


def omega(t, V, U, tau, mu):
    return -tau * np.log(granc_z_part_func(t, V, U, tau, mu))


def entropy(t, V, U, tau, mu):
    energies = energies_concat(t, V, U, mu)
    z = granc_z_part_func(t, V, U, tau, mu)
    numerator = np.sum(
        np.array([np.exp(1 / tau * np.sum(np.delete(energies, i))) * energies[i] for i in range(len(energies))]))
    denominator = np.sum(np.array([np.exp(1 / tau * np.sum(np.delete(energies, i))) for i in range(len(energies))]))
    return np.log(z) + 1 / tau * numerator / denominator


def num_part_func(t, V, U, tau, mu):
    z = granc_z_part_func(t, V, U, tau, mu)
    return tau / z * np.sum(
        np.array([i / tau * np.exp(i / tau) * can_z_part_func(t, V, U, tau, i) for i in range(0, 3)]))


