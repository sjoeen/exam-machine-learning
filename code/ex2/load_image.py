import numpy as np
import matplotlib.pyplot as plt


frey_faces = np.loadtxt('frey-faces.csv', delimiter=' ')
random_indices = np.random.choice(frey_faces.shape[0], 5, replace=False)
    #5 random images


plt.figure(figsize=(10, 5))
for i, idx in enumerate(random_indices):
    img = frey_faces[idx].reshape(28, 20)  
    plt.subplot(1, 5, i + 1)
    plt.axis('off')  
    plt.imshow(img, cmap='gray')


plt.show()


