import numpy as np
import matplotlib.pyplot as plt
from metrics import FR_Index
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_rand_score
import random

from datasets import selectDataset
from FCM import FCM
from MFCM import MFCM
from filters import *

def experiment(indexData, mc, nRep, numVar):

	exec_time = 0

	listaMetricas = []

	synthetic = selectDataset(indexData)
	dataset = synthetic[0]
	ref = synthetic[1]
	nClusters = synthetic[2]
	dataName = synthetic[3]

	for i in range(mc):

		nObj = len(dataset)
		print(f'Num var: {len(dataset[0])}')

		centersAll = np.zeros((nRep, nClusters))

		for c in range(nRep):
			centersAll[c] = random.sample(range(1, nObj), nClusters)

		Jmin = 2147483647     	# int_max do R
		bestL = 0				# melhor valor de L_resp
		bestM = 0				# melhor valor de M_resp

		for r in range(nRep):
			centers = list(map(int, centersAll[r,].tolist()))
	
			resp = MFCM(dataset, centers, 2)
	
			J = resp[0]
			L_resp = resp[1]
			M_resp = resp[2]
	
			if (Jmin > J):
				Jmin = J
				bestL = L_resp
				bestM = M_resp
	
			exec_time += resp[4]
			print(f'MC: {i + 1}, Rep: {r + 1}')

	resultado_filtro = variance_filter(dataset, bestM, nClusters)
	metricas = calculate_accuracy(bestL, ref, bestM)
	listaMetricas.append(metricas)
	print(f'\nMC {i + 1}: \nFR Index: {metricas[0]}\nARI: {metricas[1]}%\nF1 Score: {metricas[2]}%')

	resultado_filtro[0].sort(key=lambda k : k[0])

	x_var = resultado_filtro[0][-1][1]
	y_var = resultado_filtro[0][-2][1]

	print(f'Eixos: {x_var}, {y_var}')
	x_axis = dataset[:, x_var] 
	y_axis = dataset[:, y_var]

	plot_results(x_axis, y_axis, bestL, ref, dataName, exec_time, bestM)

	dataset = apply_filter(dataset, resultado_filtro, numVar)

	# print(dataset)

	Jmin = 2147483647     	# int_max do R
	bestL = 0				# melhor valor de L_resp
	bestM = 0				# melhor valor de M_resp

	for r in range(nRep):
		centers = list(map(int, centersAll[r,].tolist()))

		resp = MFCM(dataset, centers, 2)

		J = resp[0]
		L_resp = resp[1]
		M_resp = resp[2]

		if (Jmin > J):
			Jmin = J
			bestL = L_resp
			bestM = M_resp

		exec_time += resp[4]
		print(f'MC: {i + 1}, Rep: {r + 1}')

	# resultado_filtro = variance_filter(dataset, bestM, nClusters)
	metricas = calculate_accuracy(bestL, ref, bestM)
	listaMetricas.append(metricas)
	print(f'\nMC {i + 1}: \nFR Index: {metricas[0]}\nARI: {metricas[1]}%\nF1 Score: {metricas[2]}%')

	resultado_filtro[0].sort(key=lambda k : k[0])

	# Plotando os resultados

	x_var = resultado_filtro[0][-1][1]
	y_var = resultado_filtro[0][-2][1]

	print(f'Eixos: {x_var}, {y_var}')
	x_axis = dataset[:, x_var]  
	y_axis = dataset[:, y_var]

	# x_axis = dataset[:, 0]  
	# y_axis = dataset[:, 1]

	print(f'\n{resultado_filtro[1]}: ')					
	for item in range(len(resultado_filtro[0])):
		print(f'var{item + 1}: {resultado_filtro[0][item]}')

	plot_results(x_axis, y_axis, bestL, ref, dataName, exec_time, bestM)

	return [J, bestL, exec_time, bestM]

def calculate_accuracy(L, ref, U):
	ari = adjusted_rand_score(L, ref) * 100
	f1_score_result = f1_score(y_true=ref, y_pred=L,average='micro') * 100
	fr = FR_Index(U, ref)

	return [fr, ari, f1_score_result]

def plot_results(x_axis, y_axis, L, ref, dataset_name, exec_time, U):
	fig, (result, original) = plt.subplots(1, 2)

	fig.subplots_adjust(bottom=0.25)
	
	result.scatter(x_axis, y_axis, c=L)
	result.set_title("MFCM - " + dataset_name)
	
	accuracy = calculate_accuracy(L, ref, U)
	result.set(xlabel="\nFR Index: {:.2f}\nARI: {:.2f}%\nF1 Score: {:.2f}%\nTempo de Execução: {:.2f}s".format(accuracy[0],accuracy[1], accuracy[2], exec_time))

	original.scatter(x_axis, y_axis, c=ref)
	original.set_title("Referência")

	plt.show()

# definindo a função main no python
if __name__ == "__main__":
	mc = 1
	nRep = 100

	result = experiment(12, mc, nRep, 1)
