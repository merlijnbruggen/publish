import numpy as np
from scipy.integrate import solve_ivp

# Define the Van der Pol oscillator equations
def vanderpol(t, xy, mu):
    x, y = xy
    dxdt = y
    dydt = mu * (1 - x ** 2) * y - x
    return [dxdt, dydt]

# Parameters for the Van der Pol oscillator
mu = 0.5

# Initial conditions
xy0 = [1.0, 1.0]

# Time points for simulation
t_span = (0, 1000)
t_eval = np.linspace(*t_span, 1000)

# Simulate the Van der Pol oscillator
sol = solve_ivp(vanderpol, t_span, xy0, args=(mu,), t_eval=t_eval)

# Discretize the state space (consider both x-coordinate and y-coordinate)
x = sol.y[0]
y = sol.y[1]
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)
num_bins = 3  # Number of bins for discretization along x-coordinate and y-coordinate
x_discrete = np.digitize(x, np.linspace(-10, 10, num_bins), right=True)
y_discrete = np.digitize(y, np.linspace(-10, 10, num_bins), right=True)

# Build the transition matrix
transition_matrix = np.zeros((num_bins, num_bins, num_bins, num_bins))

# Loop over time steps to count transitions
for i in range(len(sol.t) - 1):
    state_current_x = x_discrete[i]
    state_current_y = y_discrete[i]
    state_next_x = x_discrete[i + 1]
    state_next_y = y_discrete[i + 1]
    
    # Ensure indices are within bounds
    state_next_x = np.clip(state_next_x, 0, num_bins - 1)
    state_next_y = np.clip(state_next_y, 0, num_bins - 1)
    
    transition_matrix[state_current_x, state_current_y, state_next_x, state_next_y] += 1

# Normalize transition probabilities
transition_matrix_smoothed = transition_matrix + 1e-6
transition_matrix /= np.sum(transition_matrix_smoothed, axis=(2, 3), keepdims=True)




transition_matrix_2d = transition_matrix.reshape(num_bins**2, num_bins**2)

print(transition_matrix_2d)
print(transition_matrix_2d.shape)


# Visualize transition matrix
import matplotlib.pyplot as plt


# Plot oscillator
plt.plot(x,y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Van der Pol Oscillator")
plt.show()


# Plot transition matrix
plt.imshow(transition_matrix_2d, cmap='viridis', interpolation='nearest')
plt.xlabel('Next x bin')
plt.ylabel('Current x bin')
plt.title('Transition Probabilities for Van der Pol Oscillator')
plt.colorbar(label='Probability')
plt.show()


