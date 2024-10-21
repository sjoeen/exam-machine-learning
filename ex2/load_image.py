import numpy as np
import matplotlib.pyplot as plt

# Load dataset (assuming each row is a flattened image)
frey_faces = np.loadtxt('frey-faces.csv', delimiter=' ')

# Get 5 random images
random_indices = np.random.choice(frey_faces.shape[0], 5, replace=False)

# Plot the 5 randomly selected images
plt.figure(figsize=(10, 5))
for i, idx in enumerate(random_indices):
    img = frey_faces[idx].reshape(28, 20)  # Reshape each row into a 28x20 image
    plt.subplot(1, 5, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')  # Hide axes for a cleaner look

plt.show()
