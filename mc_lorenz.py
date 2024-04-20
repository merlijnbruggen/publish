import numpy as np
from scipy.integrate import solve_ivp

# Define the Lorenz system equations
def lorenz(t, xyz, sigma, rho, beta):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parameters for the Lorenz system
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# Initial conditions
xyz0 = [1.0, 1.0, 1.0]

# Time points for simulation
t_span = (0, 1000)
t_eval = np.linspace(*t_span, 100)

# Simulate the Lorenz system
sol = solve_ivp(lorenz, t_span, xyz0, args=(sigma, rho, beta), t_eval=t_eval)

# Discretize the state space (consider all three coordinates)
xyz = sol.y
xyz_min, xyz_max = np.min(xyz, axis=1), np.max(xyz, axis=1)
num_bins = 5  # Number of bins for discretization along each coordinate
xyz_discrete = np.array([np.digitize(coord, np.linspace(min_val, max_val, num_bins)) 
                         for coord, min_val, max_val in zip(xyz, xyz_min, xyz_max)])

# Build the transition matrix
transition_matrix = np.zeros((num_bins, num_bins, num_bins, num_bins, num_bins, num_bins))

# Loop over time steps to count transitions
for i in range(len(sol.t) - 1):
    state_current = tuple(xyz_discrete[:, i])
    state_next = tuple(xyz_discrete[:, i + 1])
    # Ensure indices are within bounds
    state_current = tuple(np.clip(state_current, 0, num_bins - 1))
    state_next = tuple(np.clip(state_next, 0, num_bins - 1))
    transition_matrix[state_current + state_next] += 1

# Normalize transition probabilities
transition_matrix_smoothed = transition_matrix + 1e-6
transition_matrix /= np.sum(transition_matrix_smoothed, axis=(3, 4, 5), keepdims=True)

# Flatten the transition matrix
transition_matrix_2d = transition_matrix.reshape(num_bins**3, num_bins**3)

print(transition_matrix_2d)
print(transition_matrix_2d.shape)

'''
# Visualize transition matrix
import matplotlib.pyplot as plt

plt.imshow(transition_matrix_2d, cmap='viridis', interpolation='nearest')
plt.xlabel('Next state bin')
plt.ylabel('Current state bin')
plt.title('Transition Probabilities for Lorenz System')
plt.colorbar(label='Probability')
plt.show()
'''



def construct_markov_chain(transition_matrix, num_steps):
    num_states = transition_matrix.shape[0]
    current_state = np.random.randint(num_states)  # Start from a random initial state
    markov_chain = [current_state]  # Initialize the Markov chain with the initial state
    
    for _ in range(num_steps - 1):
        # Get the probability distribution for transitioning from the current state
        transition_probs = transition_matrix[current_state]
        
        # Normalize the probability distribution to ensure it sums up to 1
        transition_probs /= np.sum(transition_probs)
        
        # Sample the next state based on the normalized probability distribution
        next_state = np.random.choice(num_states, p=transition_probs)
        
        # Update the current state
        current_state = next_state
        
        # Add the next state to the Markov chain
        markov_chain.append(current_state)
    
    return markov_chain


construct_markov_chain(transition_matrix_2d, 100)