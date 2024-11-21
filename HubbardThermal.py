import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.optimize import root_scalar
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d


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


def compute_terms(energies_1, energies_2, values_1, values_2, tau, mu, U):
    """
    Computes the weighted terms for a given property (density, kinetic, vee).
    """
    term1p = np.dot(values_1, np.exp(-1 / tau * (energies_1 - mu)))
    term2p = np.dot(values_2, np.exp(-1 / tau * (energies_2 - 2 * mu)))
    term3p = np.dot(values_1, np.exp(-1 / tau * (energies_1 + U - 3 * mu)))
    return term1p, term2p, term3p


def property_calculation(t, V, U, tau, mu, operator_1p, operator_2p, diagonalize_h1=True, diagonalize_h2=True):
    """
    Generalized function to calculate a property (density, kinetic, vee).
    Parameters `operator_1p` and `operator_2p` are functions that compute the required diagonal values.
    """
    # Compute eigenvalues and eigenvectors
    (energies_1, eigenvectors_1), (energies_2, eigenvectors_2) = compute_eigenvalues_and_vectors(t, V, U)

    # Compute diagonal values
    values_1 = operator_1p(eigenvectors_1, t, V, U) if diagonalize_h1 else np.zeros_like(energies_1)
    values_2 = operator_2p(eigenvectors_2, t, V, U) if diagonalize_h2 else np.zeros_like(energies_2)

    # Compute the partition function
    z = granc_z_part_func(t, V, U, tau, mu)

    # Compute terms
    term1p, term2p, term3p = compute_terms(energies_1, energies_2, values_1, values_2, tau, mu, U)

    # Compute final property
    return 1 / z * (term1p + term2p + term3p)


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


def kinetic(t, V, U, tau, mu):
    def operator_1p(eigenvectors, t, V, U):
        return (eigenvectors.T @ h_1p(t, 0) @ eigenvectors).diagonal()

    def operator_2p(eigenvectors, t, V, U):
        return (eigenvectors.T @ h_2p(t, 0, 0) @ eigenvectors).diagonal()

    return property_calculation(t, V, U, tau, mu, operator_1p, operator_2p)


def vee(t, V, U, tau, mu):
    def operator_1p(eigenvectors, t, V, U):
        return (eigenvectors.T @ h_1p(0, 0) @ eigenvectors).diagonal()

    def operator_2p(eigenvectors, t, V, U):
        return (eigenvectors.T @ h_2p(0, 0, U) @ eigenvectors).diagonal()

    return property_calculation(t, V, U, tau, mu, operator_1p, operator_2p)


def target_function(t, V, U, tau, mu, nn):
    return gran_can_A(t, V, U, tau, mu) - (V / 2) * nn



def delta_v_of_rho_list(t, U, tau, mu):
    # Generate delta_v values from 0 to 130 with a step of 0.05
    delta_v_values = np.arange(0, 130.05, 0.05)
    # Compute the real part of rho and -delta_v for each delta_v
    data = [(np.real(density(t, delta_v, U, tau, mu)), -delta_v) for delta_v in delta_v_values]
    return np.array(data)  # Convert to a NumPy array for easier processing

# Interpolation function
def delta_v_of_rho(t, U, tau, mu):
    # Get the data list
    data = delta_v_of_rho_list(t, U, tau, mu)
    # Separate the data into x (real part of rho) and y (-delta_v)
    x = data[:, 0]  # Real part of rho
    y = data[:, 1]  # -delta_v
    # Create and return the interpolating function
    return interp1d(x, y, kind='linear', fill_value="extrapolate")


# Define reusable functions
def plot_function(function, tau_values, v_space, x_label, y_label, legend_title, title=None):
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
    for tau in tau_values:
        plt.plot(v_space, func_vectorized(0.5, v_space, 1, tau, 0.5), label=r"$\tau=$" + f"{tau}")
    if title:
        plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title=legend_title)
    plt.grid(True)
    plt.show()


def plot_interpolated_function(interpolating_function_factory, tau_values, x_range, t, U, mu, x_label, y_label,
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
    for tau in tau_values:
        interpolating_function = interpolating_function_factory(t, U, tau, mu)
        y_values = interpolating_function(x_range)
        plt.plot(x_range, y_values, label=f"tau = {tau}")
    if title:
        plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title=legend_title)
    plt.grid(True)
    plt.ylim(-20,1)
    #plt.show()


# Parameters
tau_values = [1.0, 2.0, 3.0, 4.0, 5.0]
v_space = np.linspace(0, 80, 50)
t, U, mu = 0.5, 1.0, 0.5
x_range = np.linspace(0, 2, 500)

# Example usage
# plot_function(density, tau_values, v_space, x_label=r"$-\Delta v$", y_label="Density", legend_title="Legend:")
# plot_function(kinetic, tau_values, v_space, x_label=r"$-\Delta v$", y_label="Kinetic", legend_title="Legend:")
# plot_function(vee, tau_values, v_space, x_label=r"$-\Delta v$", y_label="Vee", legend_title="Legend:")


tau_values = [1.0]
plot_interpolated_function(
    delta_v_of_rho, tau_values, x_range, t, U, mu,
    x_label=r"Re[$\rho$]", y_label=r"$-\Delta v$", legend_title="Legend:",
    title=r"Interpolated Function $\Delta v$ vs $\rho$ for Different $\tau$"
)
# Parameters
t = 0.5    # Replace with actual parameter
U = 1.0    # Replace with actual parameter
tau = 1.0  # Replace with actual parameter
mu = 0.5  # Replace with actual parameter
nn_values = np.linspace(0, 2, 500)  # Range of nn values from 0 to 2

# Compute the optimal V for each nn
v_max_values = []
for nn in nn_values:
    # Maximization using minimize_scalar
    result = minimize_scalar(lambda V: -target_function(t, V, U, tau, mu, nn))
    v_max_values.append(result.x)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(nn_values, v_max_values, label='Optimal V', color='blue')
plt.xlabel("Nn")
plt.ylabel("Optimal V")
plt.title("Optimal V as a Function of Nn")
plt.legend()
plt.grid(True)
plt.ylim(-20,1)
plt.show()