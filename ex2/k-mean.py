import numpy as np
from typing import Optional,Tuple
import matplotlib.pyplot as plt

class Kmean:
    
    def __init__(self,data:np.array,measure,k:int,tolerance=10e-3):

        self.data = data
        self.k = k
        self.tolerance = tolerance
        self.measure = measure
        self.cluster_assignments = {}

        """
        creation of centroids using random samples, uses samples because of data
        complexity.
        """
        self.num_samples = data.shape[0]
        random_row = np.random.choice(self.num_samples, size=k, replace=False)
        self.centroids = data[random_row, :]

    def euclidean(self,cord_1,cord_2):

        return np.sqrt(np.sum((np.array(cord_1) - np.array(cord_2))**2))
    
    def manhatten(self,cord_1,cord_2):

        return np.sum(np.abs(np.array(cord_1) - np.array(cord_2)))

    
    def fit(self,max_iter):
        """
        fit function follows steps given in 2.2
        """

        for _ in range(max_iter):
            argmin_list = self._distance()

            self._update_centroids(argmin_list)
                #update all datapoint into closest centroid

            if self._check():
                #check if the centorids have moved less than a given tolerance
                self._store_cluster_assignments(argmin_list)
                break


        return self._store_cluster_assignments(argmin_list)
        

    def _distance(self)-> np.array:
        """
        calulate distance between all the centroids and each datapoint
        return the index of closest centroid for each sample in a np list
        """

        num_samples = self.data.shape[0]
        num_centroids = self.centroids.shape[0]
        
 
        distances = np.zeros((num_samples, num_centroids))
            #makes an empty matrix (samples x centroids)


        for i in range(num_samples):
            for j in range(num_centroids):
                distances[i, j] = self.measure(self.data[i], self.centroids[j])
                    #calulate distance, between centroid and sample

        return np.argmin(distances,axis=1)
    
    
    def _update_centroids(self,argmin_list: np.array):
        """
        update all datapoint into closest centroid,also saves the prevoius centroids.
        """
        self.previous_centroids = self.centroids.copy()


        for i in range(self.k):
            points_in_cluster = self.data[argmin_list == i]
            self.centroids[i] = np.mean(points_in_cluster, axis=0)


    def _check(self):
        """
        This function checks if there is no signicant changes between updating
        of the centroids.
        """
        if self.measure(self.previous_centroids,self.centroids) < self.tolerance:
            return True
        
        return False
    
    def _store_cluster_assignments(self, argmin_list):
        """
        save all the datapoints, only used when the function ends
        """

        for i in range(self.k):
            self.cluster_assignments[i] = []
                #empty list for each centroid
            
        for i in range(self.num_samples):
            centroid = argmin_list[i]  
                # Get the index of the closest centroid for data point 
            self.cluster_assignments[centroid].append(self.data[i])
                #store the data into the centroid

        return self.cluster_assignments


    

if __name__ == '__main__':

    frey_faces = np.loadtxt('frey-faces.csv', delimiter=' ')
    k = int(input("select amount of centroids: "))

    kmeans = Kmean(data=frey_faces, measure=None, k=k)
    kmeans.measure = kmeans.manhatten
        #create model 

    cluster_assignments = kmeans.fit(100)
    centroids = kmeans.centroids
        #get values of centroid/data


    figure, axises = plt.subplots(k, 6, figsize=(8, 8))

    """
    chatgpt made the display loop:
    https://chatgpt.com/share/671583ab-a288-800d-9671-c556a342f36f
    """
    for i in range(k):  
        for j in range(6):  
                # Reshape the image from 1D (560,) to 2D (28, 20)
            image = np.reshape(kmeans.cluster_assignments[i][j], (28, 20))
            axises[i, j].imshow(image, cmap='gray')  # Plot the image
            axises[i, j].axis('off')  
                # removes axis values for each picture.

    plt.show()



