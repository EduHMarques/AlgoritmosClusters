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

		exec_time += resp[4]
		print(f'\nMC: {mc_i + 1}, Rep: {r + 1}')
		
	dict = {'Jmin': Jmin, 'bestL': bestL, 'bestM': bestM, 'exec_time': exec_time}
	
	return dict

def experiment(indexData, mc, nRep, numVar):

	## Inicializando variáveis
	exec_time = 0
	result = {}

	listaMetricas = []

	synthetic = selectDataset(indexData)
	dataset = synthetic[0]
	ref = synthetic[1]
	nClusters = synthetic[2]
	dataName = synthetic[3]

	## Monte Carlo
	for i in range(mc):

		nObj = len(dataset)
		print(f'Num var: {len(dataset[0])}')

		centersAll = np.zeros((nRep, nClusters))

		for c in range(nRep):
			centersAll[c] = random.sample(range(1, nObj), nClusters)

		result = execute(nRep, dataset, centersAll, i, exec_time)

	## Obtendo resultados
	resultado_filtro = variance_filter(dataset, result['bestM'], nClusters)
	metricas = calculate_accuracy(result['bestL'], ref, result['bestM'])
	listaMetricas.append(metricas)
	print(f'\nMC {i + 1}: \nFR Index: {metricas[0]}\nARI: {metricas[1]}%\nF1 Score: {metricas[2]}%')

	## Gerando primeiro plot
	resultado_filtro[0].sort(key=lambda k : k[0])

	x_var = resultado_filtro[0][-1][1]
	y_var = resultado_filtro[0][-2][1]

	print(f'Eixos: {x_var}, {y_var}')
	x_axis = dataset[:, x_var] 
	y_axis = dataset[:, y_var]

	plot1 = [x_axis, y_axis, result['bestL']]
	# plot_results(x_axis, y_axis, result['bestL'], ref, dataName, exec_time, result['bestM'])

	## Aplicando filtro
	dataset = apply_filter(dataset, resultado_filtro, numVar)
	result = execute(nRep, dataset, centersAll, 0, 0)

	## Obtendo resultados
	resultado_filtro = variance_filter(dataset, result['bestM'], nClusters)
	# metricas = calculate_accuracy(result['bestL'], ref, result['bestM'])
	# listaMetricas.append(metricas)
	print(f'\nFR Index: {metricas[0]}\nARI: {metricas[1]}%\nF1 Score: {metricas[2]}%')

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

	plot2 = [x_axis, y_axis, result['bestL']]

	## Plot final
	plot_results(plot1, plot2, ref, dataName, 0, 0)

	# return [J, bestL, exec_time, bestM]

def calculate_accuracy(L, ref, U):
	ari = adjusted_rand_score(L, ref) * 100
	f1_score_result = f1_score(y_true=ref, y_pred=L,average='micro') * 100
	fr = FR_Index(U, ref)

	return [fr, ari, f1_score_result]

def plot_results(p1, p2, ref, dataset_name, exec_time, U):
	fig, axes = plt.subplots(2, 2, figsize=(10, 8))

	# fig.subplots_adjust(bottom=0.25)
	
	# Define as posições do plot em variáveis
	ax_top_left = axes[0, 0]
	ax_top_right = axes[0, 1]
	ax_bottom = axes[1, 0]

	# Plota resultados
	ax_top_left.scatter(p1[0], p1[1], c=ref, label='Referência')
	ax_top_right.scatter(p1[0], p1[1], c=p1[2], label=f'{dataset_name} - Sem Filtro')
	ax_bottom.scatter(p2[0], p2[1], c=p2[2], label=f'{dataset_name} - Com Filtro')

	# Adiciona títulos aos subplots
	ax_top_left.set_title('Referência', weight='bold')
	ax_top_right.set_title(f'{dataset_name} - Sem Filtro', weight='bold')
	ax_bottom.set_title(f'{dataset_name} - Com Filtro', weight='bold')

	# Remove o gráfico no canto inferior direito
	axes[1, 1].remove()

	# Ajusta os espaçamentos entre os subplots
	plt.tight_layout()
	
	# accuracy = calculate_accuracy(L, ref, U)
	# result.set(xlabel="\nFR Index: {:.2f}\nARI: {:.2f}%\nF1 Score: {:.2f}%\nTempo de Execução: {:.2f}s".format(accuracy[0],accuracy[1], accuracy[2], exec_time))

	plt.show()

# definindo a função main no python
if __name__ == "__main__":
	mc = 1
	nRep = 10

	result = experiment(7, mc, nRep, 5)