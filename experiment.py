import numpy as np
import matplotlib.pyplot as plt
from metrics import FR_Index
from sklearn.metrics import f1_score, davies_bouldin_score, silhouette_score, normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import random

from datasets import selectDataset
from FCM import FCM
from MFCM import MFCM
from filters import *

def execute(nRep, dataset, centersAll, mc_i, exec_time):
	Jmin = 2147483647	# int_max do R
	bestL = 0			# melhor valor de L_resp
	bestM = 0			# melhor valor de M_resp

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
			# print(f'Conv criterio: {Jmin}')

		exec_time += resp[4]
		print(f'MC: {mc_i + 1}, Rep: {r + 1}')
		
	dict = {'Jmin': Jmin, 'bestL': bestL, 'bestM': bestM, 'exec_time': exec_time}
	
	return dict

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
		print(f'Num var: {len(dataset[0])}')

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

	print(f'\nMC {i + 1}: \nARI: {metricas[0]}\nNMI: {metricas[1]}%\nSilhouette: {metricas[2]}%\nDB: {metricas[3]}')

	# print(f'\nMC {i + 1}: \nARI: {metricas[0]}\nNMI: {metricas[1]}%\nSilhouette: {metricas[2]}%\nDB: {metricas[3]}')

	## Gerando primeiro plot
	resultado_filtro[0].sort(key=lambda k : k[0])

	x_var = resultado_filtro[0][-1][1]
	y_var = resultado_filtro[0][-2][1]

	print(f'Eixos: {x_var}, {y_var}')
	x_axis = dataset[:, x_var] 
	y_axis = dataset[:, y_var]

	plot1 = [x_axis, y_axis, result['bestL'], result['exec_time']]

	## Aplicando filtro
	dataset_antigo = dataset
	dataset = apply_filter(dataset, resultado_filtro, numVar)
	result = execute(nRep, dataset, centersAll, 0, 0)

	## Obtendo resultados
	resultado_filtro = variance_filter(dataset, result['bestM'], nClusters)
	# metricas = calculate_accuracy(result['bestL'], ref, result['bestM'])
	# listaMetricas.append(metricas)
	print(f'\nARI: {metricas[0]}\nNMI: {metricas[1]}%\nSilhouette: {metricas[2]}%\nDB: {metricas[3]}')

	## Gerando segundo plot
	resultado_filtro[0].sort(key=lambda k : k[0])

	x_var = resultado_filtro[0][-1][1]
	y_var = resultado_filtro[0][-2][1]

	print(f'Eixos: {x_var}, {y_var}')
	x_axis = dataset[:, x_var]  
	y_axis = dataset[:, y_var]

	print(f'\n{resultado_filtro[1]}: ')					
	for item in range(len(resultado_filtro[0])):
		print(f'var{item + 1}: {resultado_filtro[0][item]}')

	plot2 = [x_axis, y_axis, result['bestL'], result['exec_time']]

	## Plot final
	plot_results(plot1, plot2, ref, dataName, 0, 0, dataset, dataset_antigo)

	# return [J, bestL, exec_time, bestM]

def calculate_accuracy(L, ref, U, dataset):
	ari = adjusted_rand_score(L, ref) * 100
	nmi = normalized_mutual_info_score(ref, L)
	silhouette = silhouette_score(dataset, L)
	db = davies_bouldin_score(dataset, L)

	return [ari, nmi, silhouette, db]

def plot_results(p1, p2, ref, dataset_name, exec_time, U, dataset, dataset_antigo):
	fig, axes = plt.subplots(2, 2, figsize=(10, 8))

	# fig.subplots_adjust(bottom=0.25)
	
	# Define as posições do plot em variáveis
	ax_top_left = axes[0, 0]
	ax_top_right = axes[0, 1]
	ax_bottom = axes[1, 0]
	ax_info = axes[1, 1]

	# Plota resultados
	ax_top_left.scatter(p1[0], p1[1], c=ref, label='Referência')
	ax_top_right.scatter(p1[0], p1[1], c=p1[2], label=f'{dataset_name} - Sem Filtro')
	ax_bottom.scatter(p2[0], p2[1], c=p2[2], label=f'{dataset_name} - Com Filtro')
	# ax_info

	# Adiciona títulos aos subplots
	ax_top_left.set_title('Referência', weight='bold')
	ax_top_right.set_title(f'{dataset_name} - Sem Filtro', weight='bold')
	ax_bottom.set_title(f'{dataset_name} - Com Filtro', weight='bold')

	# Remove o gráfico no canto inferior direito e adiciona métricas
	acc1 = calculate_accuracy(p1[2], ref, U, dataset_antigo)
	acc2 = calculate_accuracy(p2[2], ref, U, dataset)
	
	metrics_info1 = ("Resultado sem filtro:\nARI: {:.2f}%\nNMI: {:.2f}\nSilhoutte: {:.2f}\nDB: {:.2f}\n\nTempo de Execução: {:.2f}s".format(acc1[0], acc1[1], acc1[2], acc1[3], p1[3]))
	metrics_info2 = ("Resultado com filtro:\nARI: {:.2f}%\nNMI: {:.2f}\nSilhoutte: {:.2f}\nDB: {:.2f}\n\nTempo de Execução: {:.2f}s".format(acc2[0], acc2[1], acc2[2], acc2[3], p2[3]))
	
	# Ajusta os espaçamentos entre os subplots
	ax_info.text(0, 0.75, metrics_info1, va='center', fontsize=11)
	ax_info.text(0, 0.25, metrics_info2, va='center', fontsize=11)
	
	ax_info.axis('off')

	plt.tight_layout()

	plt.show()

# definindo a função main no python
if __name__ == "__main__":
	mc = 3
	nRep = 50

	result = experiment(15, mc, nRep, 3)