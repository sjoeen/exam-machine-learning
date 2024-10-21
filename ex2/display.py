import numpy as np
import matplotlib.pyplot as plt

# Coordinates for two points
point_A = np.array([0, 0])
point_B = np.array([10, 5])

# Create the plot
fig, ax = plt.subplots()

# Plot the points
ax.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], 'ro', label='Points A and B')

# Plot Manhattan distance (moving along grid lines)
ax.plot([point_A[0], point_B[0]], [point_A[1], point_A[1]], 'b--', label='Manhattan Distance')
ax.plot([point_B[0], point_B[0]], [point_A[1], point_B[1]], 'b--')

# Plot Euclidean distance (direct diagonal line)
ax.plot([point_A[0], point_B[0]], [point_A[1], point_B[1]], 'g-', label='Euclidean Distance')

# Set the labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

# Add titles
ax.set_title('Comparison of Manhattan and Euclidean Distances')

plt.grid(True)
plt.show()
