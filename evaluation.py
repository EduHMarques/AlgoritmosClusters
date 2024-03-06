import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as sklearnKMeans
from classic_methods.dash2002 import Dash2002
from classic_methods.mitra2002 import feature_selection
from experiment import *
from datasets import selectDataset

def evaluate(indexData, pVar, mc, nRep, seed):
    ## Loading Data

    log = ''
    
    dataset, ref, nclusters, dataset_name = selectDataset(indexData)
    n_samples, n_features = dataset.shape

    numVar = int(pVar * n_features)

    log += f'#################### {dataset_name} ####################\n\n'
    log += f'Seed: {seed} | numVar: {numVar} ({pVar*100}%)\n'
    log += f'Dataset: {dataset_name} | n_samples: {n_samples} | n_features: {n_features} | n_clusters: {nclusters}\n'

    ## MFCM
    
    log += '\nMFCM\n'
    log += f'MC: {mc} | nRep: {nRep}\n'

    result, mfcm_time, centers = exec_mfcm(indexData, mc, nRep)
    dataset_novo = run_filter(dataset, result, ref, numVar, nclusters)

    ## Dash2002

    # threshold = 0.5 * n_features
    threshold = numVar

    log += '\nDash2002\n'
    log += f'threshold: {threshold}\n'

    model = Dash2002(dataset, threshold)

    entropy = model.execute(dataset)
    dash_features = model.forward_selection()

    log += f"Entropy: {entropy}\n"
    # log += (f"Variáveis selecionadas: {dash_features}")

    ## Mitra2002

    log += ('\nMitra2002\n')
    log += f'K: {int(threshold)}\n'

    mitra_features = feature_selection(dataset, k=int(threshold))
    # log += (f"Variáveis selecionadas: {mitra_features}")

    ## Feature selection
    X_dash = dataset[:, dash_features]
    X_mitra = dataset[:, mitra_features]

    ## K Means
    K = nclusters

    # KMeansresult, KMeanstime = exec_kmeans(K, nRep, dataset_novo, centers) # KMeans com filtro MFCM
    KMeansresult = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
    KMeansresult.fit(dataset_novo)
    KMeansresult = KMeansresult.labels_

    og_Kmeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
    og_Kmeans.fit(dataset)
    y_pred_og = og_Kmeans.labels_

    mitra_Kmeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
    mitra_Kmeans.fit(X_mitra)
    y_pred1 = mitra_Kmeans.labels_

    dash_Kmeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
    dash_Kmeans.fit(X_dash)
    y_pred2 = dash_Kmeans.labels_

    ## Metrics

    log += '\nResults:\n\n'

    log += 'KMeans sem filtro:\n'
    results = calculate_accuracy(y_pred_og, ref, None, dataset)
    log += (f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}\n')

    log += '\nMFCM:\n'
    results = calculate_accuracy(KMeansresult, ref, None, dataset_novo)
    log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {mfcm_time}s\n'

    log += '\nDash2002:\n'
    results = calculate_accuracy(y_pred2, ref, None, X_dash)
    log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}\n'

    log += '\nMitra2002:\n'
    results = calculate_accuracy(y_pred1, ref, None, X_mitra)
    log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}\n\n'

    return log

    ## Plotting
    # plt.figure(figsize=(18, 5))

    # plt.subplot(1, 5, 1)
    # plt.scatter(dataset[:,0], dataset[:,1], c=ref)
    # plt.title('Original')

    # plt.subplot(1, 5, 2)
    # plt.scatter(dataset[:,0], dataset[:,1], c=y_pred_og)
    # plt.title('KMeans sem filtro')

    # plt.subplot(1, 5, 3)
    # plt.scatter(dataset_novo[:,0], dataset_novo[:,1], c=filtered_result)
    # plt.title('Filtro por Variância - MFCM')

    # plt.subplot(1, 5, 4)
    # plt.scatter(X_mitra[:,0], X_mitra[:,1], c=y_pred1)
    # plt.title('Mitra2002')

    # plt.subplot(1, 5, 5)
    # plt.scatter(X_dash[:,0], X_dash[:,1], c=y_pred2)
    # plt.title('Dash2002')

    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    log_file = open('AlgoritmosClusters/logs/evaluation_classic_methods.txt', 'a', newline='\n')

    SEED = 42
    nRep = 100
    
    datasets = [17]
    pVars = [0.25, 0.50]

    for d in datasets:
        for p in pVars:
            log = evaluate(d, p, 1, nRep, SEED)
            print(log)
            log_file.write(log)

    print("---------> Done <---------")
    log_file.close()