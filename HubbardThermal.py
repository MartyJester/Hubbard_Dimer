import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.optimize import root_scalar
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


# def density(t, V, U, tau, mu):
#     n12 = np.diag([2, 1, 1, 0, 1, 1])
#     n22 = np.diag([0, 1, 1, 2, 1, 1])
#     n11 = np.diag([1, 1, 0, 0])
#     n21 = np.diag([0, 0, 1, 1])
#     Dn2OP = n12 - n22
#     Dn1OP = n11 - n21
#     energies_1, eigenvectors_1 = np.linalg.eig(h_1p(t, V))
#     energies_2, eigenvectors_2 = np.linalg.eig(h_2p(t, V, U))
#     rho_1 = eigenvectors_1.T @ Dn1OP @ eigenvectors_1
#     rho_1 = rho_1.diagonal()
#     rho_2 = eigenvectors_2.T @ Dn2OP @ eigenvectors_2
#     rho_2 = rho_2.diagonal()
#     z = granc_z_part_func(t, V, U, tau, mu)
#     term1p = np.dot(rho_1, np.exp(-1 / tau * (energies_1 - mu)))
#     term2p = np.dot(rho_2, np.exp(-1 / tau * (energies_2 - 2 * mu)))
#     term3p = np.dot(rho_1, np.exp(-1 / tau * (energies_1 + U - 3 * mu)))
#     dens = 1 / z * (term1p + term2p + term3p)
#     return dens
#
#
# def kinetic(t, V, U, tau, mu):
#     energies_1, eigenvectors_1 = np.linalg.eig(h_1p(t, V))
#     energies_2, eigenvectors_2 = np.linalg.eig(h_2p(t, V, U))
#     T_1 = eigenvectors_1.T @ h_1p(t, 0) @ eigenvectors_1
#     T_1 = T_1.diagonal()
#     T_2 = eigenvectors_2.T @ h_2p(t, 0, 0) @ eigenvectors_2
#     T_2 = T_2.diagonal()
#     z = granc_z_part_func(t, V, U, tau, mu)
#     term1p = np.dot(T_1, np.exp(-1 / tau * (energies_1 - mu)))
#     term2p = np.dot(T_2, np.exp(-1 / tau * (energies_2 - 2 * mu)))
#     term3p = np.dot(T_1, np.exp(-1 / tau * (energies_1 + U - 3 * mu)))
#     T = 1 / z * (term1p + term2p + term3p)
#     return T
#
# def vee(t, V, U, tau, mu):
#     energies_1, eigenvectors_1 = np.linalg.eig(h_1p(t, V))
#     energies_2, eigenvectors_2 = np.linalg.eig(h_2p(t, V, U))
#     V_1 = eigenvectors_1.T @ h_1p(0, 0) @ eigenvectors_1
#     V_1 = V_1.diagonal()
#     V_2 = eigenvectors_2.T @ h_2p(0, 0, U) @ eigenvectors_2
#     V_2 = V_2.diagonal()
#     z = granc_z_part_func(t, V, U, tau, mu)
#     term1p = np.dot(V_1, np.exp(-1 / tau * (energies_1 - mu)))
#     term2p = np.dot(V_2, np.exp(-1 / tau * (energies_2 - 2 * mu)))
#     term3p = np.dot(V_1 + U, np.exp(-1 / tau * (energies_1 + U - 3 * mu)))
#     vee = 1 / z * (term1p + term2p + term3p)
#     return vee


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

# Example tau values
tau_values = [1.0, 2.0, 3.0, 4.0, 5.0]
v_space = np.linspace(0, 80, 50)
rho_vec = np.vectorize(density)
plt.figure(figsize=(10, 6))
for temp in tau_values:
    plt.plot(v_space, rho_vec(0.5, v_space, 1, temp, 0.5), label=r"$\tau=$"+f"{temp}")
plt.xlabel(r"$-\Delta v$")
plt.legend()
plt.ylabel('Density')
plt.grid(True)
plt.show()

tau_values = [1.0, 2.0, 3.0, 4.0, 5.0]
kinetic_vec = np.vectorize(kinetic)
plt.figure(figsize=(10, 6))
for temp in tau_values:
    plt.plot(v_space, kinetic_vec(0.5, v_space, 1, temp, 0.5), label=r"$\tau=$"+f"{temp}")
plt.xlabel(r"$-\Delta v$")
plt.legend(title="Legend:")
plt.ylabel('kin')
plt.grid(True)
plt.show()

tau_values = [1.0, 2.0, 3.0, 4.0, 5.0]
vee_vec = np.vectorize(vee)
plt.figure(figsize=(10, 6))
for temp in tau_values:
    plt.plot(v_space, vee_vec(0.5, v_space, 1, temp, 0.5), label=r"$\tau=$"+f"{temp}")
plt.xlabel(r"$-\Delta v$")
plt.legend(title="Legend:")
plt.ylabel('vee')
plt.grid(True)
plt.show()



# Other parameters
t, U, mu = 1.0, 2.0, 1.

# Initialize the plot
plt.figure(figsize=(10, 6))

# Loop over tau values
for tau in tau_values:
    # Create the interpolation function for the current tau
    interpolating_function = delta_v_of_rho(t, U, tau, mu)

    # Generate x values for plotting
    x_values = np.linspace(0, 2, 500)  # Adjust the range as needed
    y_values = interpolating_function(x_values)

    # Plot with a label for the current tau
    plt.plot(x_values, y_values, label=f"tau = {tau}")

# Finalize the plot
plt.title(r"Interpolated Function $\Delta v$ vs $\rho$ for Different $\tau$")
plt.xlabel(r"Re[$\rho$]")
plt.ylabel(r"$-\Delta v$")
plt.legend(title="Legend:")
plt.grid(True)
plt.show()