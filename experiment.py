import numpy as np
import pandas as pd
import random

from FCM import FCM

def experiment(dataset, mc, nRep, nClusters):
  nObj = len(dataset)
  print('nObj: ', nObj)
  print('nClusters: ', nClusters)

  centersAll = np.zeros((nRep, nClusters))

  for i in range(nRep):
    centersAll[i] = random.sample(range(1, nObj), nClusters)

  # print(centersAll)

  Jmin = 2147483647     # int_max do R
  partMin = 0

  for i in range(nRep):
    centers = list(map(int, centersAll[i,].tolist()))
    # print(f'Rep({i}) - Centros: {centers}')
    resp = FCM(dataset, centers, 2)

    J = resp[0]
    L_resp = resp[1]
    M_resp = resp[2]

    if (Jmin > J):
      Jmin = J
      partMin = L_resp
    
  print('Jmin: ', J)
  print('Matriz: ', L_resp)


dataset = pd.read_csv("iris.txt", sep=",")
dataset.columns = ['A', 'B', 'C', 'D', 'E']
 
dataset_unlabeled = dataset.drop('E', axis=1)
dataset_ref = dataset["E"].tolist()
# print(dataset_ref)
 
dataset_unlabeled = dataset_unlabeled.to_numpy()
# print(dataset_unlabeled)

# nObj = len(dataset_unlabeled)

experiment(dataset_unlabeled, 1, 100, 3)