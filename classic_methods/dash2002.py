import numpy as np
from sklearn.datasets import load_iris, load_wine

def euclidean_distance(xi, xj):
  return np.sqrt(np.sum(np.square(xj-xi)))

def normalize(x, xmin, xmax):
  return (x-xmin)/(xmax-xmin)

def normalized_euclidean_distance(data):
  n_samples = data.shape[0]

  distances = []
  for i in range(n_samples-1):
    for j in range(i+1, n_samples):
      if data.ndim == 1:
        distances.append(euclidean_distance(data[i], data[j]))
      else:
        distances.append(euclidean_distance(data[i, :], data[j, :]))

  distances = np.sort(distances)
  xmin = np.min(distances)
  xmax = np.max(distances)

  norm_distances = []
  for dist in distances:
    norm_distances.append(normalize(dist, xmin, xmax))

  return norm_distances

class Dash2002:

  def __init__(self, data, threshold, beta=10, b=100):
    self.data = data
    self.threshold = threshold
    self.beta = beta
    self.b = b

  def execute(self, data):
      distances = normalized_euclidean_distance(data)
      buckets = [0 for _ in range(self.b+1)]

      for dist in distances:
        index = int(dist*self.b)
        buckets[index] += 1

      calculated_entropy = self.entropy(distances)

      return round(calculated_entropy, 4)

  def calc_mu(self, distances):
    Qmin = 0.05 * len(distances)

    Bi = 0
    dist = 0
    for i in range(15):
      if distances[i] > Qmin and distances[i] > distances[Bi]:
        Bi = i
        dist = distances[i]

    Et = 0.02
    mu = (1/self.beta) * np.log((np.exp(self.beta * dist) + np.exp(0)) / Et - 1)

    return mu

  def compute_eij(self, distances, dist):
    mu = self.calc_mu(distances)

    if mu >= dist:
      eij = ( np.exp(self.beta*dist) - np.exp(0) ) / ( np.exp(self.beta*mu) - np.exp(0) )
    else:
      eij = ( np.exp(self.beta*(1-dist)) - np.exp(0) ) / ( np.exp(self.beta*(1-mu)) - np.exp(0) )

    return eij

  def entropy(self, distances):
    e = 0

    for dist in distances:
        e += self.compute_eij(distances, dist)

    return e

  def forward_selection(self):
    e = 100000000
    index = 0
    n_samples, n_features = self.data.shape
    selected_vars = []

    # Select best feature
    for i in range(n_features):
      result = self.execute(self.data[:, i])
      if e > result:
        e = result
        index = i

    selected_vars.append(index)

    # Select remaining features
    while len(selected_vars) < self.threshold:
      remaining_vars = [i for i in range(n_features) if i not in selected_vars]
      e = 100000000
      index = 0

      for i in remaining_vars:
        result = self.execute(self.data[:, selected_vars+[i]])
        if e > result:
          e = result
          index = i

      selected_vars.append(index)

    return selected_vars
  

if __name__ == '__main__':
    data, target = load_wine(return_X_y=True)
    threshold = 0.5 * data.shape[1]
    model = Dash2002(data, threshold)

    entropy_r = model.execute(data)
    selected_vars = model.forward_selection()

    print(f"Entropia do dataset: {entropy_r}")

    print("\nVariáveis selecionadas:")
    for i in selected_vars:
        print(f'Variável {i}')
    print(f"\nThreshold: {int(threshold)}")
