import numpy as np
import pandas as pd
import random

from FCM import FCM

def experiment(dataset, mc, nRep, nClusters):
  nObj = len(dataset)
  # print('nObj: ', nObj)
  # print('nClusters: ', nClusters)

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
    
  print(f"\nRESULTADO:\n\nJmin: {J}\n\nMatriz L:{L_resp}")


def selectDataset(id):
	if id == 1:
		# Iris
		
		dataset = pd.read_csv("PIBIC/iris.txt", sep=",", header=None)
		dataset_ref = dataset.iloc[:,-1].tolist()

		dataset_unlabeled = dataset.drop(dataset.columns[-1], axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
			
		nClusters = 3
		
		return [dataset_unlabeled, dataset_ref, nClusters]
	elif id == 2:
		# Glass

		dataset = pd.read_csv("./datasets/glass.txt", sep=",", header=None)
		dataset_ref = dataset.iloc[:,-1].tolist()
		
		columns = [dataset.columns[0], dataset.columns[-1]]

		dataset_unlabeled = dataset.drop(columns, axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
		
		nClusters = 6
		
		return [dataset_unlabeled, dataset_ref, nClusters]
	elif id == 3:
		# Dataset Sintético

		n1 = 5
		n2 = 5
		n3 = 5
		
		n = n1 + n2 + n3		
		
		nClusters = 3
		
		mu_11 = 0
		mu_12 = 16
		
		mu_21 = -8
		mu_22 = 8
		
		mu_31 = -16
		mu_32 = -5
		
		sigma_11 = pow(6, 0.5)
		sigma_12 = pow(6, 0.5)
		
		sigma_21 = pow(13, 0.5)
		sigma_22 = pow(13, 0.5)
		
		sigma_31 = pow(20, 0.5)
		sigma_32 = pow(20, 0.5)

		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		
		x3 = np.random.normal(mu_31, sigma_31, n3)
		y3 = np.random.normal(mu_32, sigma_32, n3)

		class1 = np.column_stack((x1, y1))
		class2 = np.column_stack((x2, y2))
		class3 = np.column_stack((x3, y3))

		synthetic = np.vstack((class1, class2, class3))

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		refClass3 = np.repeat(3, n3)
		
		ref = np.concatenate((refClass1, refClass2, refClass3))

		return [synthetic, ref, nClusters]


# definindo a função main no python
if __name__ == "__main__":
	params = selectDataset(1)
	mc = 50
	nRep = 100

	experiment(params[0], 50, 100, params[2])