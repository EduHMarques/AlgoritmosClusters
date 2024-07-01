import os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

current_dir = os.getcwd()

scaler = StandardScaler()

# Buscar dataset Artigo LFSC - 2007

def selectDataset(id):
	if id == 1:
		# Automobile
		pass
	elif id == 4:
		# Heart Statlog
		path = os.path.join(current_dir, "datasets/heart-statlog.dat")
		dataset = pd.read_csv(path, sep=" ", header=None)
		dataset_ref = [x - 1 for x in dataset.iloc[:,-1].tolist()]
		
		columns = [dataset.columns[-1]]

		dataset_unlabeled = dataset.drop(columns, axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()

		nClusters = 2
		
		dataset_unlabeled = normalize(dataset_unlabeled)

		return [dataset_unlabeled, dataset_ref, nClusters, "Heart Statlog Dataset"]
	elif id == 7:
		# Liver Disorders
		path = os.path.join(current_dir, "datasets/liver-disorders.data")
		dataset = pd.read_csv(path, sep=",", header=None)
		dataset_ref = [x - 1 for x in dataset.iloc[:,-1].tolist()]	# Ref?
		
		columns = [dataset.columns[-1]]

		dataset_unlabeled = dataset.drop(columns, axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()

		nClusters = 2
		
		dataset_unlabeled = normalize(dataset_unlabeled)

		return [dataset_unlabeled, dataset_ref, nClusters, "Liver Disorders Dataset"]
	elif id == 9:
		# Lymphography
		path = os.path.join(current_dir, "datasets/lymphography.data")
		dataset = pd.read_csv(path, sep=",", header=None)
		dataset_ref = [x - 1 for x in dataset.iloc[:,0].tolist()]
		
		columns = [dataset.columns[0]]

		dataset_unlabeled = dataset.drop(columns, axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()

		nClusters = 4
		
		dataset_unlabeled = normalize(dataset_unlabeled)

		return [dataset_unlabeled, dataset_ref, nClusters, "Lymphography Dataset"]
	elif id == 11:
		# Sonar
		path = os.path.join(current_dir, "datasets/sonar.data")
		dataset = pd.read_csv(path, sep=",", header=None)
		dataset_ref = [0 if x == 'R' else 1 for x in dataset.iloc[:,-1].tolist()]
		
		columns = [dataset.columns[-1]]
		dataset_unlabeled = dataset.drop(columns, axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()

		nClusters = 2
		
		dataset_unlabeled = normalize(dataset_unlabeled)

		return [dataset_unlabeled, dataset_ref, nClusters, "Sonar Dataset"]
	elif id == 13:
		# WDBC
		path = os.path.join(current_dir, "datasets/wdbc.data")
		dataset = pd.read_csv(path, sep=",", header=None)
		dataset_ref = [0 if x == 'M' else 1 for x in dataset.iloc[:,1].tolist()]
		
		columns = dataset.columns[[0, 1]]
		dataset_unlabeled = dataset.drop(columns, axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()

		nClusters = 2
		
		dataset_unlabeled = normalize(dataset_unlabeled)

		return [dataset_unlabeled, dataset_ref, nClusters, "Breast Cancer Wisconsin (Diagnostic) Dataset"]
	elif id == 14:
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
	if id == 16:
		# Iris
		
		path = os.path.join(current_dir, "datasets/iris.txt")
		dataset = pd.read_csv(path, sep=",", header=None)
		dataset_ref = dataset.iloc[:,-1].tolist()

		dataset_unlabeled = dataset.drop(dataset.columns[-1], axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
			
		nClusters = 3

		dataset_unlabeled = normalize(dataset_unlabeled)
		
		return [dataset_unlabeled, dataset_ref, nClusters, "Iris Dataset"]
		
	elif id == 17:
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
	elif id == 18:
		# KC2
		
		dataset = pd.read_csv("datasets/kc2.txt", sep=",", header=None)
		dataset_ref = dataset.iloc[:,-1].tolist()

		dataset_unlabeled = dataset.drop(dataset.columns[-1], axis=1)
		dataset_unlabeled = dataset_unlabeled.to_numpy()
			
		nClusters = 2

		dataset_unlabeled = normalize(dataset_unlabeled)
		
		return [dataset_unlabeled, dataset_ref, nClusters, "KC2 Dataset"]
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

		y_class1 = np.ones(int(n/3)) * 0
		y_class2 = np.ones(int(n/3)) * 1
		y_class3 = np.ones(int(n/3)) * 2
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
			[0, 0, 0],
			[0, 20, 0],
			[0, 40, 0]
		])
		sigma = np.array([
			[5, 5, 5],
			[5, 5, 5],
			[5, 5, 5]
		])
		sigma = [np.diag(sigma[i]) for i in range(3)]

		X_linear = np.random.multivariate_normal(mu[0], sigma[0], size=int(n/nClusters))
		y_linear = np.repeat(0, int(n/nClusters))
		for i in range(1, nClusters):
			X_linear = np.vstack((X_linear, np.random.multivariate_normal(mu[i], sigma[i], size=int(n/nClusters))))
			y_linear = np.hstack((y_linear, np.repeat(i, int(n/nClusters))))

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