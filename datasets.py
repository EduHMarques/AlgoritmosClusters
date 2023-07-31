import numpy as np
import pandas as pd

def selectDataset(id):
	if id == 1:
		# Iris
		
		dataset = pd.read_csv("datasets/iris.txt", sep=",", header=None)
		dataset_ref = dataset.iloc[:,-1].tolist()

		dataset_unlabeled = dataset.drop(dataset.columns[-1], axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
			
		nClusters = 3

		dataset_unlabeled = normalize(dataset_unlabeled)
		
		return [dataset_unlabeled, dataset_ref, nClusters, "Iris Dataset"]
	elif id == 2:
		# Glass

		dataset = pd.read_csv("./datasets/glass.txt", sep=",", header=None)
		dataset_ref = dataset.iloc[:,-1].tolist()
		
		columns = [dataset.columns[0], dataset.columns[-1]]

		dataset_unlabeled = dataset.drop(columns, axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
		
		nClusters = 6
		
		return [dataset_unlabeled, dataset_ref, nClusters, "Glass Dataset"]
	elif id == 3:
		# Dataset Sintético

		n1 = 100
		n2 = 100
		n3 = 100
		
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

		return [synthetic, ref, nClusters, "Dataset Sintético"]
	elif id == 4:
		# Dataset Sintético

		n1 = 50
		n2 = 50
		n3 = 50
		n4 = 50
		
		n = n1 + n2 + n3 + n4
		
		nClusters = 4
		
		mu_11 = 40
		mu_12 = 7

		mu_21 = 70
		mu_22 = 37

		mu_31 = 40
		mu_32 = 70

		mu_41 = 10
		mu_42 = 37
		
		sigma_11 = 10
		sigma_12 = 4
		
		sigma_21 = 4
		sigma_22 = 10
		
		sigma_31 = 10
		sigma_32 = 4

		sigma_41 = 4
		sigma_42 = 10

		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		
		x3 = np.random.normal(mu_31, sigma_31, n3)
		y3 = np.random.normal(mu_32, sigma_32, n3)

		x4 = np.random.normal(mu_41, sigma_41, n4)
		y4 = np.random.normal(mu_42, sigma_42, n4)

		class1 = np.column_stack((x1, y1))
		class2 = np.column_stack((x2, y2))
		class3 = np.column_stack((x3, y3))
		class4 = np.column_stack((x4, y4))

		synthetic = np.vstack((class1, class2, class3, class4))

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		refClass3 = np.repeat(3, n3)
		refClass4 = np.repeat(4, n4)
		
		ref = np.concatenate((refClass1, refClass2, refClass3, refClass4))

		return [synthetic, ref, nClusters, "Dataset Sintético 2"]

	elif id == 5:
		# (Dataset 1 do 'pimentel2013')
		n1 = 200
		n2 = 100
		n3 = 50
		
		n = n1 + n2 + n3 
		
		nClusters = 3
		
		mu_11 = 5
		mu_12 = 0

		mu_21 = 15
		mu_22 = 5

		mu_31 = 18
		mu_32 = 14

		sigma_11 = 9
		sigma_12 = 3
		
		sigma_21 = 3
		sigma_22 = 10
		
		sigma_31 = 5
		sigma_32 = 6


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

		return [synthetic, ref, nClusters, "Dataset Sintético 3"] 
	
	elif id == 6:
		# (Dataset 2 do 'pimentel2013')
		n1 = 200
		n2 = 100
		n3 = 50
		
		n = n1 + n2 + n3 
		
		nClusters = 3
		
		mu_11 = 0
		mu_12 = 0

		mu_21 = 30
		mu_22 = 0

		mu_31 = 15
		mu_32 = -3

		
		sigma_11 = 10
		sigma_12 = 2
		
		sigma_21 = 10
		sigma_22 = 2
		
		sigma_31 = 10
		sigma_32 = 2


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

		return [synthetic, ref, nClusters, "Dataset Sintético 4"]
        
	
	elif id == 7:
		# KC2
		
		dataset = pd.read_csv("datasets/kc2.txt", sep=",", header=None)
		dataset_ref = dataset.iloc[:,-1].tolist()

		dataset_unlabeled = dataset.drop(dataset.columns[-1], axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
			
		nClusters = 2

		dataset_unlabeled = normalize(dataset_unlabeled)
		
		return [dataset_unlabeled, dataset_ref, nClusters, "KC2 Dataset"]
	elif id == 8:
		# Dataset Sintético Correlacionado (gambiarra)

		n1 = 50
		n2 = 50
		
		n = n1 + n2
		
		nClusters = 2
		
		mu_11 = 0
		mu_12 = 50

		mu_21 = 30
		mu_22 = 0
		
		sigma_11 = 30
		sigma_12 = 1
		
		sigma_21 = 30
		sigma_22 = 1
		
		x1 = []
		y1 = []

		for i in range(n1):
			x1.append(np.random.normal(mu_11, sigma_11, 1) + i) 
		for i in range(n1):
			y1.append(np.random.normal(mu_12, sigma_12, 1) + i) 

		x2 = []
		y2 = []

		for i in range(n1):
			x1.append(np.random.normal(mu_21, sigma_21, 1) + i) 

		for i in range(n1):
			y1.append(np.random.normal(mu_22, sigma_22, 1) + i) 

		class1 = np.column_stack((x1, y1))
		class2 = np.column_stack((x2, y2))

		synthetic = np.vstack((class1, class2))

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		
		ref = np.concatenate((refClass1, refClass2))

		return [synthetic, ref, nClusters, "Dataset Sintético 6"]
	
	elif id == 10:
		# Dataset Sintético

		n1 = 50
		n2 = 50
		
		n = n1 + n2
		
		nClusters = 2
		
		mu_11 = 0
		mu_12 = 100

		mu_21 = 0
		mu_22 = 0
		
		sigma_11 = 20
		sigma_12 = 2
		
		sigma_21 = 20
		sigma_22 = 2
		
		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		

		class1 = np.column_stack((x1, y1))
		class2 = np.column_stack((x2, y2))

		synthetic = np.vstack((class1, class2))

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		
		ref = np.concatenate((refClass1, refClass2))

		return [synthetic, ref, nClusters, "2 elipses HOR."]
	
	elif id == 11:
		# Dataset Sintético

		n1 = 50
		n2 = 50
		
		n = n1 + n2
		
		nClusters = 2
		
		mu_11 = 0
		mu_12 = 0

		mu_21 = 50
		mu_22 = 0
		
		sigma_11 = 4
		sigma_12 = 20
		
		sigma_21 = 4
		sigma_22 = 20
		
		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		

		# class1 = np.column_stack((x1, y1))
		# class2 = np.column_stack((x2, y2))

		synthetic = np.vstack((class1, class2))

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		
		ref = np.concatenate((refClass1, refClass2))

		return [synthetic, ref, nClusters, "2 elipses VER."]
	
	elif id == 12:
		# Dataset Sintético

		n1 = 50
		n2 = 50
		
		n = n1 + n2
		
		nClusters = 2
		
		mu_11 = 0
		mu_12 = 0
		mu_13 = 0

		mu_21 = 20
		mu_22 = 0
		mu_23 = 50
		
		sigma_11 = 4
		sigma_12 = 20
		sigma_13 = 4
		
		sigma_21 = 4
		sigma_22 = 20
		sigma_23 = 4
		
		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		z1 = np.random.normal(mu_13, sigma_13, n1)
		
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		z2 = np.random.normal(mu_23, sigma_23, n2)

		class1 = np.column_stack((x1, y1, z1))
		class2 = np.column_stack((x2, y2, z2))
		
		# class1 = np.column_stack((x1, z1))
		# class2 = np.column_stack((x2, z2))

		synthetic = np.vstack((class1, class2))

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		
		ref = np.concatenate((refClass1, refClass2))

		return [synthetic, ref, nClusters, "2 elipses 3D"]
	

def normalize(dataset):
	nRows = dataset.shape[0]
	nCol = dataset.shape[1]

	normalizedArray = np.arange(nRows * nCol)
	normalizedArray = normalizedArray.reshape((nRows, nCol))
	normalizedArray = (np.zeros_like(normalizedArray)).astype('float64')

	media = np.mean(dataset, axis=0)
	dp = np.std(dataset, axis=0)

	novo = np.arange(nRows * nCol)
	novo = normalizedArray.reshape((nRows, nCol))
	novo = (np.zeros_like(novo)).astype('float64')

	for i in range (0, nRows):
		for j in range (0, nCol):
			novo[i, j] = (dataset[i, j] - media[j]) / dp[j]

	return novo

def normalize2(dataset):
	nRows = dataset.shape[0]
	nCol = dataset.shape[1]

	novo = np.arange(nRows * nCol)
	novo = novo.reshape((nRows, nCol))
	novo = (np.zeros_like(novo)).astype('float64')

	for i in range (0, nRows):
		for j in range (0, nCol):
			min = np.min(dataset[:,j])
			max = np.max(dataset[:,j])
			novo[i, j] = (dataset[i, j] - min) / (max - min)

	return novo