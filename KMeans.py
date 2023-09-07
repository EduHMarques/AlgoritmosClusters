import numpy as np
from timeit import default_timer as timer

np.random.seed(42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1-x2)))

class KMeans:
    
    def __init__(self, K=5, nRep=100):
        self.K = K
        self.nRep = nRep

        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []

    def predict(self, data):
        start = timer()

        self.data = data
        self.nObj, self.nVar = data.shape

        ## inicialização dos centroids

        # indices_amostras: lista de tamanho K que armazena valores entre 0 e nObj
        # replace = false para evitar valores (índices) repetidos
        indices_amostras = np.random.choice(self.nObj, self.K, replace=False)
        self.centroids = [self.data[ind] for ind in indices_amostras]

        # Algoritmo
        for _ in range(self.nRep):
            # Atualiza os clusters
            self.clusters = self.update_clusters(self.centroids)

            # Atualiza os centroids
            c_old = self.centroids
            self.centroids = self.get_centroids(self.clusters)

            # Checa se o algoritmo convergeu 
            if self.converged(c_old, self.centroids):
                break

        end = timer()

        return (self.get_clusters_labels(self.clusters), end-start)

    def get_clusters_labels(self, clusters):
        labels = np.empty(self.nObj)

        for ind, cluster in enumerate(clusters):
            for obj_ind in cluster:
                labels[obj_ind] = ind

        return labels

    def update_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]

        # Adiciona os objetos aos clusters com base no centroid mais próximo
        for ind, obj in enumerate(self.data):
            centroid_id = self.find_centroid(obj, centroids)
            clusters[centroid_id].append(ind)

        return clusters
    
    def find_centroid(self, obj, centroids):
        distances = [euclidean_distance(obj, centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)

        return closest_centroid
    
    def get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.nVar))

        for ind, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.data[cluster], axis=0) # Acessa apenas os objetos que se encontram no cluster atual
            centroids[ind] = cluster_mean

        return centroids
    
    def converged(self, c_old, c_new):
        distances = [euclidean_distance(c_old[i], c_new[i]) for i in range(self.K)]

        return sum(distances) == 0