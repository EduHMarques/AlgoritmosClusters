import numpy as np
import matplotlib.pyplot as plt
from metrics import FR_Index
from sklearn.metrics import f1_score, davies_bouldin_score, silhouette_score, normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import random
import time

from datasets import selectDataset
from FCM import FCM
from MFCM import MFCM
from KMeans import KMeans
from filters import *

SEED = 45
np.random.seed(SEED)
random.seed(SEED)

def execute(nRep, dataset, centersAll, mc_i, exec_time):
	Jmin = 2147483647	# int_max do R
	bestL = 0			# melhor valor de L_resp
	bestM = 0			# melhor valor de M_resp
	best_centers = 0

	for r in range(nRep):
		print(f'Rep: {r+1}/{nRep}')
		centers = list(map(int, centersAll[r,].tolist()))

		resp = MFCM(dataset, centers, 2)

		J = resp[0]
		L_resp = resp[1]
		M_resp = resp[2]

		if (Jmin > J):
			Jmin = J
			bestL = L_resp
			bestM = M_resp
			best_centers = centers

		exec_time += resp[4]
		# print(f'MC: {mc_i + 1}, Rep: {r + 1}')
		
	dict = {'Jmin': Jmin, 'bestL': bestL, 'bestM': bestM, 'exec_time': exec_time, 'best_centers': centersAll}
	# Retorna os centers de todas as iterações para o KMeans (mudar para criar uma nova lista exclusiva para o KMeans)
	
	return dict

def exec_mfcm(indexData, mc, nRep):
	
	## Inicializando variáveis
	exec_time = 0
	result = {}
	Jmin = 2147483647
	centers = 0

	## Monte Carlo
	for i in range(mc):
		SEED = int(time.time())
		np.random.seed(SEED)
		random.seed(SEED)

		synthetic = selectDataset(indexData)
		dataset = synthetic[0]
		ref = synthetic[1]
		nClusters = synthetic[2]
		dataName = synthetic[3]

		centersMC = np.zeros((nRep, nClusters))

		for c in range(nRep):
			centersMC[c] = random.sample(range(1, len(dataset)), nClusters)

		clustering = execute(nRep, dataset, centersMC, i, exec_time)
		exec_time += clustering['exec_time']

		if clustering['Jmin'] < Jmin:
			Jmin = clustering['Jmin']
			result = clustering
			centers = clustering['best_centers']
	
	return (result, exec_time, centers)

def exec_kmeans(K, nRep, dataset, centers):

	bestResult = 0
	Jmin = 2147483647

	for r in range(nRep):
		centersRep = list(map(int, centers[r,].tolist()))
		k = KMeans(K, nRep,)
		result, time, J = k.predict(dataset, SEED, centersRep)

		if J < Jmin:
			# print(f'Jmin atualizado: {J}')
			Jmin = J
			bestResult = result

	return [bestResult, time]
	
def run_filter(method, dataset, result, ref, numVar, numClusters):
		
	## Obtendo resultados
	if method == 1:
		resultado_filtro = variance_filter(dataset, result['bestM'], numClusters)
	else:
		resultado_filtro = sum_filter(dataset, result['bestM'], numClusters)
	## Aplicando filtro
	dataset = apply_filter(dataset, resultado_filtro, numVar, method)

	return dataset

def experiment(indexData, mc, nRep, numVar):

	## Inicializando variáveis
	exec_time = 0
	result = {}

	listaMetricas = []

	## Monte Carlo
	Jmin = 2147483647
	centersAll = []
	for i in range(mc):
		synthetic = selectDataset(indexData)
		dataset = synthetic[0]
		ref = synthetic[1]
		nClusters = synthetic[2]
		dataName = synthetic[3]

		nObj = len(dataset)
		# print(f'Num var: {len(dataset[0])}')

		centersMC = np.zeros((nRep, nClusters))

		for c in range(nRep):
			centersMC[c] = random.sample(range(1, nObj), nClusters)

		clustering = execute(nRep, dataset, centersMC, i, exec_time)

		if clustering['Jmin'] < Jmin:
			Jmin = clustering['Jmin']
			result = clustering
			centersAll = centersMC

	## Obtendo resultados
	resultado_filtro = variance_filter(dataset, result['bestM'], nClusters)
	metricas = calculate_accuracy(result['bestL'], ref, result['bestM'], dataset)
	listaMetricas.append(metricas)

	# print(f'\nMC {i + 1}: \nARI: {metricas[0]}\nNMI: {metricas[1]}%\nSilhouette: {metricas[2]}%\nDB: {metricas[3]}')

	## Gerando primeiro plot
	resultado_filtro[0].sort(key=lambda k : k[0])

	x_var = resultado_filtro[0][-1][1]
	y_var = resultado_filtro[0][-2][1]

	# print(f'Eixos: {x_var}, {y_var}')
	x_axis = dataset[:, x_var] 
	y_axis = dataset[:, y_var]

	plot1 = [x_axis, y_axis, result['bestL'], result['exec_time']]

	## Aplicando filtro
	dataset_antigo = dataset
	dataset = apply_filter(dataset, resultado_filtro, numVar)
	result = exec_kmeans(4, nRep, ref, dataset, dataset_antigo)

def calculate_accuracy(L, ref, U, dataset):
	ari = adjusted_rand_score(L, ref)
	nmi = normalized_mutual_info_score(ref, L)
	silhouette = silhouette_score(dataset, L)
	db = davies_bouldin_score(dataset, L)

	return [ari, nmi, silhouette, db]

def plot_results(plot_info, ref, dataset_name, exec_time, U, dataset, dataset_antigo, datasetName):
	fig, axes = plt.subplots(2, 2, figsize=(10, 8))

	# fig.subplots_adjust(bottom=0.25)
	
	# Define as posições do plot em variáveis
	ax_top_left = axes[0, 0]
	ax_top_right = axes[0, 1]
	ax_bottom = axes[1, 0]
	ax_info = axes[1, 1]

	# Plota resultados
	x_axis = dataset_antigo[:, plot_info[0]]
	y_axis = dataset_antigo[:, plot_info[1]]

	ax_top_left.scatter(x_axis, y_axis, c=ref, label='Referência')
	ax_top_right.scatter(x_axis, y_axis, c=plot_info[4], label=f'{dataset_name} - Sem Filtro')
	
	x_axis = dataset[:, 0]
	y_axis = dataset[:, 1]
	ax_bottom.scatter(x_axis, y_axis, c=plot_info[5], label=f'{dataset_name} - Com Filtro')

	# Adiciona títulos aos subplots
	ax_top_left.set_title('Referência', weight='bold')
	ax_top_right.set_title(f'{dataset_name} - Sem Filtro', weight='bold')
	ax_bottom.set_title(f'{dataset_name} - Com Filtro', weight='bold')

	# Remove o gráfico no canto inferior direito e adiciona métricas
	acc1 = calculate_accuracy(plot_info[4], ref, U, dataset_antigo)
	acc2 = calculate_accuracy(plot_info[5], ref, U, dataset)
	
	metrics_info1 = ("Resultado sem filtro:\nARI: {:.2f}\nNMI: {:.2f}\nSilhoutte: {:.2f}\nDB: {:.2f}\nTempo de Execucao: {:.2f}s".format(acc1[0], acc1[1], acc1[2], acc1[3], exec_time[0]))
	metrics_info2 = ("Resultado com filtro:\nARI: {:.2f}\nNMI: {:.2f}\nSilhoutte: {:.2f}\nDB: {:.2f}\nTempo de Execucao: {:.2f}s".format(acc2[0], acc2[1], acc2[2], acc2[3], exec_time[1]))

	metrics_txt1 = [acc1[0], acc1[1], acc1[2], acc1[3], exec_time[0], datasetName]
	metrics_txt2 = [acc2[0], acc2[1], acc2[2], acc2[3], exec_time[1], datasetName]

	for i in range(len(metrics_txt1) - 1):
		metrics_txt1[i] = round(metrics_txt1[i], 2)
		metrics_txt2[i] = round(metrics_txt2[i], 2)

	atualizaTxt(f'logs/{dataset_name}.txt', metrics_txt1)
	atualizaTxt(f'logs/{dataset_name}.txt', metrics_txt2)
	atualizaTxt(f'logs/{dataset_name}.txt', '')
	
	# Ajusta os espaçamentos entre os subplots
	ax_info.text(0, 0.75, metrics_info1, va='center', fontsize=11)
	ax_info.text(0, 0.25, metrics_info2, va='center', fontsize=11)
	
	ax_info.axis('off')

	plt.tight_layout()

	plt.show()

def atualizaTxt(nome, lista):
	arquivo = open(nome, 'a')
	for i in range(len(lista)):
		arquivo.write(str(lista[i]) + ' ')
	arquivo.write('\n')
	arquivo.close()

# definindo a função main no python
if __name__ == "__main__":
	mc = 1
	nRep = 3
	indexData = 1
	numVar = 2

	# result = experiment(14, mc, nRep, 2)
	result, ref, centers = exec_mfcm(indexData, mc, nRep)
	aux = selectDataset(indexData)
	dataset, ref, nclusters, datasetName = aux[0], aux[1], aux[2], aux[3]

	dataset_novo, plot = run_filter(dataset, result, ref, numVar, nclusters)

	## KMeans
	K = nclusters
	# print('Executando KMEANS sem filtro')
	raw_result, raw_time, raw_name = exec_kmeans(K, nRep, dataset, centers)
	# print('Executando KMEANS com filtro')
	filtered_result, filtered_time, filtered_name = exec_kmeans(K, nRep, dataset_novo, centers)
	plot.append(raw_result)
	plot.append(filtered_result)

	plot_results(plot, ref, raw_name, (raw_time, filtered_time), 0, dataset_novo, dataset, datasetName)