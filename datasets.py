import os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

current_dir = os.getcwd()

scaler = StandardScaler()

def selectDataset(id):
	if id == 1:
		# Iris
		
		path = os.path.join(current_dir, "datasets/iris.txt")
		dataset = pd.read_csv(path, sep=",", header=None)
		dataset_ref = dataset.iloc[:,-1].tolist()

		dataset_unlabeled = dataset.drop(dataset.columns[-1], axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
			
		nClusters = 3

		dataset_unlabeled = normalize(dataset_unlabeled)
		
		return [dataset_unlabeled, dataset_ref, nClusters, "Iris Dataset"]
	elif id == 2:
		# Glass

		path = os.path.join(current_dir, "AlgoritmosClusters/datasets/glass.txt")
		dataset = pd.read_csv(path, sep=",", header=None)
		dataset_ref = dataset.iloc[:,-1].tolist()
		
		columns = [dataset.columns[0], dataset.columns[-1]]

		dataset_unlabeled = dataset.drop(columns, axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()

		nClusters = 6
		
		dataset_unlabeled = normalize(dataset_unlabeled)

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

		return [synthetic, ref, nClusters, "Pimentel2013 - Data 1"] 
	
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

		mu_31 = 10
		mu_32 = 25
		
		sigma_11 = 10
		sigma_12 = 10
		
		sigma_21 = 7
		sigma_22 = 7
		
		sigma_31 = 4
		sigma_32 = 4


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

		return [synthetic, ref, nClusters, "Pimentel2013 - Data 2"]
	elif id == 7:
		# (Dataset 3 do 'pimentel2013')
		n1 = 100
		n2 = 100
		n3 = 100
		
		n = n1 + n2 + n3 
		
		nClusters = 3
		
		mu_11 = 0
		mu_12 = 0

		mu_21 = 15
		mu_22 = 3

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

		return [synthetic, ref, nClusters, "Pimentel2013 - Data 3"]
	elif id == 8:
		# (Dataset 4 do 'pimentel2013')
		n1 = 150
		n2 = 100
		n3 = 50
		
		n = n1 + n2 + n3 
		
		nClusters = 3
		
		mu_11 = 0
		mu_12 = 0

		mu_21 = 15
		mu_22 = 0

		mu_31 = -15
		mu_32 = 0
		
		sigma_11 = 4
		sigma_12 = 4
		
		sigma_21 = 4
		sigma_22 = 4
		
		sigma_31 = 4
		sigma_32 = 4


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

		return [synthetic, ref, nClusters, "Pimentel2013 - Data 4"]
	
	elif id == 9:
		# KC2
		
		dataset = pd.read_csv("datasets/kc2.txt", sep=",", header=None)
		dataset_ref = dataset.iloc[:,-1].tolist()

		dataset_unlabeled = dataset.drop(dataset.columns[-1], axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
			
		nClusters = 2

		dataset_unlabeled = normalize(dataset_unlabeled)
		
		return [dataset_unlabeled, dataset_ref, nClusters, "KC2 Dataset"]
	elif id == 10:
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
	
	elif id == 11:
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
	
	elif id == 12:
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
		

		class1 = np.column_stack((x1, y1))
		class2 = np.column_stack((x2, y2))

		synthetic = np.vstack((class1, class2))

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		
		ref = np.concatenate((refClass1, refClass2))

		return [synthetic, ref, nClusters, "2 elipses VER."]
	
	elif id == 13:
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
	elif id == 14:
		# Dataset do "Localized feature selection for clustering - 2007"

		n1 = 100
		n2 = 100
		n3 = 100
		n4 = 100
		
		n = n1 + n2+ n3 + n4
		
		nClusters = 4
		
		mu_11 = 0.5 * 10
		mu_12 = -0.5 * 10
		mu_13 = 0
		mu_14 = 0

		mu_21 = -0.5 * 10
		mu_22 = -0.5 * 10
		mu_23 = 0
		mu_24 = 0

		mu_31 = 0
		mu_32 = 0.5 * 10
		mu_33 = 0.5 * 10
		mu_34 = 0

		mu_41 = 0
		mu_42 = 0.5 * 10
		mu_43 = -0.5 * 10
		mu_44 = 0
		
		sigma_11 = 0.2 * 10
		sigma_12 = 0.2 * 10
		sigma_13 = 0.6 * 10
		sigma_14 = 0.6 * 10
		
		sigma_21 = 0.2 * 10
		sigma_22 = 0.2 * 10
		sigma_23 = 0.6 * 10
		sigma_24 = 0.6 * 10

		sigma_31 = 0.6 * 10
		sigma_32 = 0.2 * 10
		sigma_33 = 0.2 * 10
		sigma_34 = 0.6 * 10

		sigma_41 = 0.6 * 10
		sigma_42 = 0.2 * 10
		sigma_43 = 0.2 * 10
		sigma_44 = 0.6 * 10
		
		x1 = np.random.normal(mu_11, sigma_11, n1)
		y1 = np.random.normal(mu_12, sigma_12, n1)
		z1 = np.random.normal(mu_13, sigma_13, n1)
		w1 = np.random.normal(mu_14, sigma_14, n1)
		
		x2 = np.random.normal(mu_21, sigma_21, n2)
		y2 = np.random.normal(mu_22, sigma_22, n2)
		z2 = np.random.normal(mu_23, sigma_23, n2)
		w2 = np.random.normal(mu_24, sigma_24, n2)

		x3 = np.random.normal(mu_31, sigma_31, n3)
		y3 = np.random.normal(mu_32, sigma_32, n3)
		z3 = np.random.normal(mu_33, sigma_33, n3)
		w3 = np.random.normal(mu_34, sigma_34, n3)

		x4 = np.random.normal(mu_41, sigma_41, n4)
		y4 = np.random.normal(mu_42, sigma_42, n4)
		z4 = np.random.normal(mu_43, sigma_43, n4)
		w4 = np.random.normal(mu_44, sigma_44, n4)

		class1 = np.column_stack((x1, y1, z1, w1))
		class2 = np.column_stack((x2, y2, z2, w2))
		class3 = np.column_stack((x3, y3, z3, w3))
		class4 = np.column_stack((x4, y4, z4, w4))
		
		# class1 = np.column_stack((x1, z1))
		# class2 = np.column_stack((x2, z2))

		synthetic = np.vstack((class1, class2, class3, class4))

		refClass1 = np.repeat(1, n1)
		refClass2 = np.repeat(2, n2)
		refClass3 = np.repeat(3, n3)
		refClass4 = np.repeat(4, n4)
		
		ref = np.concatenate((refClass1, refClass2, refClass3, refClass4))

		return [synthetic, ref, nClusters, "Artigo LFSC - 2007"]
	elif id == 15:
		# Wine

		path = os.path.join(current_dir, "AlgoritmosClusters/datasets/wine.txt")
		dataset = pd.read_csv(path, sep=",", header=None)
		dataset_ref = dataset.iloc[:,0].tolist()
		
		columns = [dataset.columns[0]]

		dataset_unlabeled = dataset.drop(columns, axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()

		nClusters = 3
		
		dataset_unlabeled = normalize(dataset_unlabeled)

		return [dataset_unlabeled, dataset_ref, nClusters, "Wine Dataset"]
	elif id == 16:
		# Balance

		dataset = pd.read_csv("./datasets/balance.txt", sep=",", header=None)
		dataset_ref = dataset.iloc[:,0].tolist()
		
		columns = [dataset.columns[0]]

		dataset_unlabeled = dataset.drop(columns, axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()

		nClusters = 3
		
		dataset_unlabeled = normalize(dataset_unlabeled)

		return [dataset_unlabeled, dataset_ref, nClusters, "Balance Dataset"]
	elif id == 17:
		# Scene | OPENML: ID 312 | 300 features
		path = os.path.join(current_dir, "datasets/scene.arff")
		data = arff.loadarff(path)
		
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)
		dataset = dataset.drop(dataset.columns[294:299], axis=1)

		dataset_ref = dataset.iloc[:,-1].tolist()

		dataset_unlabeled = dataset.drop(dataset.columns[-1], axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
			
		nClusters = 2
		dataset_unlabeled = normalize(dataset_unlabeled)
		
		return [dataset_unlabeled, dataset_ref, nClusters, "Scene Dataset"]
	elif id == 18:
		# Madelon | OPENML: ID 1485 | 500 features
		path = os.path.join(current_dir, "datasets/madelon.arff")
		data = arff.loadarff(path)
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)

		dataset_ref = dataset.iloc[:,-1].tolist()

		dataset_unlabeled = dataset.drop(dataset.columns[-1], axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
			
		nClusters = 2
		dataset_unlabeled = normalize(dataset_unlabeled)
				
		return [dataset_unlabeled, dataset_ref, nClusters, "Madelon Dataset"]
	elif id == 19:
		# Hiva Agnostic | OPENML: ID 1039 | 1000 features
		path = os.path.join(current_dir, "datasets/hiva_agnostic.arff")
		data = arff.loadarff(path)
		dataset = pd.DataFrame(data[0])
		dataset[dataset.columns[-1]] = dataset.iloc[:,-1].astype(int)

		dataset_ref = dataset.iloc[:,-1].tolist()

		dataset_unlabeled = dataset.drop(dataset.columns[-1], axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
			
		nClusters = 2
		dataset_unlabeled = normalize(dataset_unlabeled)
		
		return [dataset_unlabeled, dataset_ref, nClusters, "Hiva Agnostic Dataset"]
	elif id == 20:
		# Musk (Version 1) | UCI Machine Learning Repository | 165 features
		path = os.path.join(current_dir, "datasets/musk1.data")
		dataset = pd.read_csv(path, header=None)
		dataset = dataset.drop(dataset.columns[[0, 1, -1]], axis=1)

		dataset_ref = dataset.iloc[:,-1].tolist()

		dataset_unlabeled = dataset.drop(dataset.columns[-1], axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
			
		nClusters = 2
		dataset_unlabeled = normalize(dataset_unlabeled)
		
		# print("Dataset selecionado: Musk (Version 1)\n")
		
		return [dataset_unlabeled, dataset_ref, nClusters, "Musk (Version 1) Dataset"]
	elif id == 21:
		n = 210 
		nClusters = 3 

		mu = np.array([
			[0, 0],
			[15, 0],
			[10, 25] 
		])
		sigma = np.array([
			[1, 10],
			[5, 1],
			[3, 5] 
			])
		sigma = [np.diag(sigma[i]) for i in range(3)]

		X_class1 = np.random.multivariate_normal(mu[0], sigma[0], size=int(n/nClusters))
		X_class2 = np.random.multivariate_normal(mu[1], sigma[1], size=int(n/nClusters))
		X_class3 = np.random.multivariate_normal(mu[2], sigma[2], size=int(n/nClusters))

		X = np.vstack((X_class1, X_class2, X_class3))

		y_class1 = np.ones(int(n/3)) * 1
		y_class2 = np.ones(int(n/3)) * 2
		y_class3 = np.ones(int(n/3)) * 3
		y = np.hstack((y_class1, y_class2, y_class3))

		synthetic = X
		data_ref = y

		synthetic = scaler.fit_transform(synthetic)

		mu_str = repr(mu).replace('array', 'np.array')
		sigma_str = repr(sigma).replace('array', 'np.array')

		parameters_str = f"mu = {mu_str}\n\nsigma = {sigma_str}"

		return [synthetic, data_ref, nClusters, "Spherical Gaussian Distribution - 3 Classes", parameters_str]
	elif id == 22:
		n = 210				# Dataset Sintético Relação linear
		nClusters = 3

		mu = np.array([
			[0, 0],
			[30, 0],
			[10, 25] 
			])
		sigma = np.array([
			[1, 10],
			[5, 1],
			[3, 5] 
			])
		sigma = [np.diag(sigma[i]) for i in range(3)]

		X_linear = np.random.multivariate_normal(mu[0], sigma[0], size=int(n/nClusters))
		y_linear = np.repeat(1, int(n/nClusters))
		for i in range(1, nClusters):
			X_linear = np.vstack((X_linear, np.random.multivariate_normal(mu[i], sigma[i], size=int(n/nClusters))))
			y_linear = np.hstack((y_linear, np.repeat(i+1, int(n/nClusters))))

		synthetic = X_linear
		data_ref_ = y_linear

		synthetic = scaler.fit_transform(synthetic)

		mu_str = repr(mu).replace('array', 'np.array')
		sigma_str = repr(sigma).replace('array', 'np.array')

		parameters_str = f"mu = {mu_str}\n\nsigma = {sigma_str}"

		return [synthetic, data_ref_, nClusters, "Relacao linear", parameters_str]
	elif id == 23:
		n = 200 
		n_classes = 3

		mu1 = np.array([0, 0])
		mu2 = np.array([30, 0])
		mu3 = np.array([10, 25])

		coef = 0.5
		base = 2

		X_class1 = mu1 + coef * np.random.rand(n // n_classes, 2)
		X_class2 = mu2 + coef * (base**np.random.rand(n // n_classes, 1)) * np.random.rand(n // n_classes, 2)
		X_class3 = mu3 + coef * (base**np.random.rand(n // n_classes, 1)) * np.random.rand(n // n_classes, 2)

		synthetic = np.vstack((X_class1, X_class2, X_class3))
		synthetic = scaler.fit_transform(synthetic)

		print(synthetic)

		data_ref = np.concatenate((np.repeat(1, n // n_classes), np.repeat(2, n // n_classes), np.repeat(3, n // n_classes)))

		return [synthetic, data_ref, n_classes, "Relacao Exponencial"]
	
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