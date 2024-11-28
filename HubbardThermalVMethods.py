import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
# import jax.numpy as jnp
# from scipy.optimize import root_scalar



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
    if 0 <= n <= 4:
        return np.sum(np.exp(-1 / tau * energy_combined(t, V, U)[n]))
    else:
        print('error in can_z_part_func')


def energies_concat(t_par, V_par, U_par, mu_par):
    array_mu = np.array([0, 1, 2, 3, 4]) * mu_par
    energies = energy_combined(t_par, V_par, U_par)
    result_rows = []
    for i, row in enumerate(energies):
        subtracted_row = row - array_mu[i]
        result_rows.append(subtracted_row)
    energies = np.concatenate(result_rows)
    return energies


def granc_z_part_func(t_par, V_par, U_par, tau_par, mu_par):
    energies = energies_concat(t_par, V_par, U_par, mu_par)
    zpart = np.sum(np.exp(energies * (-1 / tau_par)))
    return zpart


def omega(t_par, V_par, U_par, tau_par, mu_par):
    return -tau_par * np.log(granc_z_part_func(t_par, V_par, U_par, tau_par, mu_par))


def entropy(t_par, V_par, U_par, tau_par, mu_par):
    energies = energies_concat(t_par, V_par, U_par, mu_par)
    z = granc_z_part_func(t_par, V_par, U_par, tau_par, mu_par)
    numerator = np.sum(
        np.array([np.exp(1 / tau_par * np.sum(np.delete(energies, i))) * energies[i] for i in range(len(energies))]))
    denominator = np.sum(np.array([np.exp(1 / tau_par * np.sum(np.delete(energies, i))) for i in range(len(energies))]))
    return np.log(z) + 1 / tau_par * numerator / denominator


def num_particles_func(t, V, U, tau, mu):
    Zg = granc_z_part_func(t, V, U, tau, mu)
    weight_n = np.sum(np.array([n * np.exp(1 / tau * mu * n) * can_z_part_func(t, V, U, tau, n) for n in range(1, 5)]))
    return weight_n / Zg


def gran_can_A(t, V, U, tau, mu):
    return mu * num_particles_func(t, V, U, tau, mu) + omega(t, V, U, tau, mu)


def compute_eigenvalues_and_vectors(t, V, U):
    """
    Computes eigenvalues and eigenvectors for h_1p and h_2p Hamiltonians.
    """
    energies_1, eigenvectors_1 = np.linalg.eig(h_1p(t, V))
    energies_2, eigenvectors_2 = np.linalg.eig(h_2p(t, V, U))
    return (energies_1, eigenvectors_1), (energies_2, eigenvectors_2)


def compute_terms(energies_1, energies_2, values_1, values_2, tau_par, mu_par, U_par, operator):
    """
    Computes the weighted terms for a given property (density, kinetic, vee).
    """
    term1p = np.dot(values_1, np.exp(-1 / tau_par * (energies_1 - mu_par)))
    term2p = np.dot(values_2, np.exp(-1 / tau_par * (energies_2 - 2 * mu_par)))
    term3p = np.dot(values_1, np.exp(-1 / tau_par * (energies_1 + U_par - 3 * mu_par)))
    term4p = 0
    if operator == 'vee':
        term3p = np.dot(values_1 + U_par, np.exp(-1 / tau_par * (energies_1 + U_par - 3 * mu_par)))
        term4p = (2 * U_par) * np.exp(-1 / tau_par *(2 * U_par - 4 * mu_par))
    return term1p, term2p, term3p, term4p


def property_calculation(t_par, V_par, U_par, tau_par, mu_par, operator_1p, operator_2p, operator, diagonalize_h1=True, diagonalize_h2=True):
    """
    Generalized function to calculate a property (density, kinetic, vee).
    Parameters `operator_1p` and `operator_2p` are functions that compute the required diagonal values.
    """
    # Compute eigenvalues and eigenvectors
    (energies_1, eigenvectors_1), (energies_2, eigenvectors_2) = compute_eigenvalues_and_vectors(t_par, V_par, U_par)

    # Compute diagonal values
    values_1 = operator_1p(eigenvectors_1, t_par, V_par, U_par) if diagonalize_h1 else np.zeros_like(energies_1)
    values_2 = operator_2p(eigenvectors_2, t_par, V_par, U_par) if diagonalize_h2 else np.zeros_like(energies_2)

    # Compute the partition function
    z = granc_z_part_func(t_par, V_par, U_par, tau_par, mu_par)

    # Compute terms
    term1p, term2p, term3p, term4p = compute_terms(energies_1, energies_2, values_1, values_2, tau_par, mu_par, U_par, operator = operator)

    # Compute final property
    return 1 / z * (term1p + term2p + term3p + term4p)


# Specific property functions

def density(t, V, U, tau, mu):
    n12 = np.diag([2, 1, 1, 0, 1, 1])
    n22 = np.diag([0, 1, 1, 2, 1, 1])
    n11 = np.diag([1, 1, 0, 0])
    n21 = np.diag([0, 0, 1, 1])
    Dn2OP = n12 - n22
    Dn1OP = n11 - n21

    def operator_1p(eigenvectors, t, V, U):
        return (eigenvectors.T @ Dn1OP @ eigenvectors).diagonal()

    def operator_2p(eigenvectors, t, V, U):
        return (eigenvectors.T @ Dn2OP @ eigenvectors).diagonal()

    return property_calculation(t, V, U, tau, mu, operator_1p, operator_2p)


def kinetic(t_par, V_par, U_par, tau_par, mu_par):
    def operator_1p(eigenvectors, t_par, V_par, U_par):
        return (eigenvectors.T @ h_1p(t_par, 0) @ eigenvectors).diagonal()

    def operator_2p(eigenvectors, t_par, V_par, U_par):
        return (eigenvectors.T @ h_2p(t_par, 0, 0) @ eigenvectors).diagonal()

    return property_calculation(t_par, V_par, U_par, tau_par, mu_par, operator_1p, operator_2p, operator = 'kin')


def vee(t_par, V_par, U_par, tau_par, mu_par):
    def operator_1p(eigenvectors, t_par, V_par, U_par):
        return (eigenvectors.T @ h_1p(0, 0) @ eigenvectors).diagonal()

    def operator_2p(eigenvectors, t_par, V_par, U_par):
        return (eigenvectors.T @ h_2p(0, 0, U_par) @ eigenvectors).diagonal()

    return property_calculation(t_par, V_par, U_par, tau_par, mu_par, operator_1p, operator_2p, operator = 'vee')


def target_function(t, V, U, tau, mu, nn):
    return gran_can_A(t, V, U, tau, mu) - (V / 2) * nn



def delta_v_of_rho_list(t, U, tau, mu):
    # Generate delta_v values from 0 to 130 with a step of 0.05
    delta_v_values = np.arange(0, 130.05, 0.05)
    # Compute the real part of rho and -delta_v for each delta_v
    data = [(np.real(density(t, delta_v, U, tau, mu)), -delta_v) for delta_v in delta_v_values]
    return np.array(data)  # Convert to a NumPy array for easier processing

# Interpolation function
def delta_v_of_rho(t_par, U_par, tau_par, mu_par):
    # Get the data list
    data = delta_v_of_rho_list(t_par, U_par, tau_par, mu_par)
    # Separate the data into x (real part of rho) and y (-delta_v)
    x = data[:, 0]  # Real part of rho
    y = data[:, 1]  # -delta_v
    # Create and return the interpolating function
    return interp1d(x, y, kind='linear', fill_value="extrapolate")

def delta_v_of_rho_kantorovich(densities, t_par, U_par, tau_par, mu_par):
    v_max_values_kant = []
    for dens in densities:
        # Maximization using minimize_scalar
        kant = minimize_scalar(lambda V_par: -target_function(t_par, V_par, U_par, tau_par, mu_par, dens))
        v_max_values_kant.append(kant.x)
    return np.array(v_max_values_kant)


# Define reusable functions
def plot_function(function, tau_values, t_par, U_par, mu_par, v_space, x_label, y_label, legend_title, title=None):
    """
    Plots the given function for different tau values.

    Args:
    - function: The function to be plotted.
    - tau_values: List of tau values to iterate over.
    - v_space: The range of x values for the plot.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - legend_title: Title for the legend.
    - title: Title for the plot (optional).
    """
    func_vectorized = np.vectorize(function)
    plt.figure(figsize=(10, 6))
    for tau_par in tau_values:
        plt.plot(v_space, func_vectorized(t_par, v_space, U_par, tau_par, mu_par), label=r"$\tau=$" + f"{tau_par}")
    if title:
        plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title=legend_title)
    plt.grid(True)
    plt.show()


def plot_interpolated_function(interpolating_function_factory, tau_values, x_range, t_par, U_par, mu_par, x_label, y_label,
                               legend_title, title=None):
    """
    Plots interpolated functions for different tau values.

    Args:
    - interpolating_function_factory: Function to generate interpolating functions.
    - tau_values: List of tau values.
    - x_range: Range of x values for the plot.
    - t: Parameter for the interpolating function.
    - U: Parameter for the interpolating function.
    - mu: Parameter for the interpolating function.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - legend_title: Title for the legend.
    - title: Title for the plot (optional).
    """
    plt.figure(figsize=(10, 6))
    for tau_par in tau_values:
        interpolating_function = interpolating_function_factory(t_par, U_par, tau_par, mu_par)
        y_values = interpolating_function(x_range)
        plt.plot(x_range, y_values, label=f"tau = {tau_par}")
    if title:
        plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title=legend_title)
    plt.grid(True)
    plt.ylim(-20,1)
    plt.show()


ti,vi,Ui,taui,mui = 1.,2,3,4,5
print(kinetic(ti,vi,Ui,taui,mui))# -0.214451
print(vee(ti,vi,Ui,taui,mui))# 2.50531
print(entropy(ti,vi,Ui,taui,mui))# 2.46694