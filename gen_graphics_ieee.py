import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns # type: ignore

# Estilo Seaborn para gráficos mais bonitos
sns.set(style="whitegrid")

# Variáveis
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
y_labels = ['Adjusted Rand Index (ARI)', 'Normalized Mutual Information (NMI)', 'Silhouette Score', 'Davies-Bouldin Score (DB)']
dataset_names = ['Hiva']
datasets = ['Hiva']

for j, dataset in enumerate(datasets):
    path = 'csv/' + dataset
    mf_m = pd.read_csv(path + '/MF_M.csv')
    mf_v = pd.read_csv(path + '/MF_V.csv')
    maxvar = pd.read_csv(path + '/MAXVAR.csv')
    ls = pd.read_csv(path + '/LS.csv')
    fsfs = pd.read_csv(path + '/FSFS.csv')
    mcfs = pd.read_csv(path + '/MCFS.csv')
    udfs = pd.read_csv(path + '/UDFS.csv')
    baseline = pd.read_csv(path + '/BASELINE.csv')
    
    methods = {'MF_M': mf_m, 'MF_V': mf_v, 'MaxVar': maxvar, 'LS': ls, 'FSFS': fsfs, 'MCFS': mcfs, 'UDFS': udfs, 'Baseline': baseline}
    colors = ['b', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray']
    markers = ['o', 'v', 's', 'p', '*', 'X', 'D', 'd']
    
    for i in range(4):  # Um gráfico por métrica
        plt.figure(figsize=(8, 6))

        for idx, (label, data) in enumerate(methods.items()):
            plt.plot(x, data.iloc[:, i], label=label, color=colors[idx], marker=markers[idx], markersize=8, linewidth=2)

        plt.xlabel('Percentage (%) of features', fontsize=12, weight='bold')
        plt.ylabel(y_labels[i], fontsize=12, weight='bold')
        plt.title(f'{dataset_names[j]} Dataset - {y_labels[i]}', fontsize=14, weight='bold')

        # Estilizando o gráfico
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True)
        plt.legend(loc='lower right', fontsize=10)  # Ajustar a posição da legenda

        # Ajustando o formato dos ticks do eixo x para ficarem em porcentagem
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # Exibir o gráfico
        plt.tight_layout()
        plt.show()
