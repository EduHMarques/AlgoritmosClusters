import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as sklearnKMeans
from literature_methods.dash2002 import Dash2002
from literature_methods.mitra2002 import feature_selection
from literature_methods.maxvar import maxVar
from literature_methods.laplacian_score import laplacian_score
from experiment import *
from datasets import selectDataset
from timeit import default_timer as timer

def evaluate(indexData, pVar, mc, nRep, seed):
    ## Loading Data

    log = ''
    
    dataset, ref, nclusters, dataset_name = selectDataset(indexData)
    n_samples, n_features = dataset.shape

    numVar = int(pVar * n_features)

    log += f'*{"-"*30}* {dataset_name} *{"-"*30}*\n\n'
    log += f'Seed: {seed} | numVar: {numVar} ({pVar*100}%)\n'
    log += f'Dataset: {dataset_name} | n_samples: {n_samples} | n_features: {n_features} | n_clusters: {nclusters}\n'
    log += 'Methods: MaxVar, Dash2002, Mitra2002'

    log += '\nParameters:\n'

    ## MFCM
    
    log += '\nMFCM\n'
    log += f'MC: {mc} | nRep: {nRep}\n'

    result, mfcm_time, centers = exec_mfcm(indexData, mc, nRep)
    dataset_novo = run_filter(dataset, result, ref, numVar, nclusters)

    ## MaxVar

    start = timer()
    maxVar_features = maxVar(dataset, pVar)
    end = timer()
    maxvar_time = end - start

    ## Laplacian Score:

    start = timer()
    LS_features = laplacian_score(dataset, pVar)
    end = timer()
    ls_time = end - start

    ## Dash2002

    # threshold = 0.5 * n_features
    threshold = numVar

    log += '\nDash2002\n'
    log += f'threshold: {threshold}\n'

    start = timer()

    model = Dash2002(dataset, threshold)

    entropy = model.execute(dataset)
    dash_features = model.forward_selection()

    end = timer()
    dash_time = end - start

    log += f"Entropy: {entropy}\n"
    # log += (f"Variáveis selecionadas: {dash_features}")

    ## Mitra2002

    log += ('\nMitra2002\n')
    log += f'K: {int(threshold)}\n'

    start = timer()
    mitra_features = feature_selection(dataset, k=int(threshold))
    # log += (f"Variáveis selecionadas: {mitra_features}")
    end = timer()
    mitra_time = end - start

    ## Feature selection
    X_maxvar = dataset[:, np.array(maxVar_features, dtype='int32')]
    X_LS = dataset[:, LS_features]
    X_dash = dataset[:, dash_features]
    X_mitra = dataset[:, mitra_features]

    ## K Means
    K = nclusters

    # KMeansresult, KMeanstime = exec_kmeans(K, nRep, dataset_novo, centers) # KMeans com filtro MFCM
    og_Kmeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
    og_Kmeans.fit(dataset)
    y_pred_og = og_Kmeans.labels_
    
    KMeansresult = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
    KMeansresult.fit(dataset_novo)
    KMeansresult = KMeansresult.labels_

    maxvar_Kmeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
    maxvar_Kmeans.fit(X_maxvar)
    y_pred0 = maxvar_Kmeans.labels_

    LS_KMeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
    LS_KMeans.fit(X_LS)
    y_pred1 = LS_KMeans.labels_

    mitra_Kmeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
    mitra_Kmeans.fit(X_mitra)
    y_pred2 = mitra_Kmeans.labels_

    dash_Kmeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
    dash_Kmeans.fit(X_dash)
    y_pred3 = dash_Kmeans.labels_

    ## Metrics

    log += '\nResults:\n\n'

    log += 'KMeans sem filtro:\n'
    results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred_og, ref, None, dataset)))
    log += (f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}\n')

    log += '\nMFCM:\n'
    results = list(map(lambda x: round(x, 8), calculate_accuracy(KMeansresult, ref, None, dataset_novo)))
    log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(mfcm_time, 4)}s\n'

    log += '\nMaxVar:\n'
    results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred0, ref, None, X_maxvar)))
    log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(maxvar_time, 4)}s\n'

    log += '\nLS:\n'
    results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred1, ref, None, X_LS)))
    log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(ls_time, 4)}s\n'

    log += '\nMitra2002:\n'
    results = list(map(lambda x: round(x, 8),calculate_accuracy(y_pred2, ref, None, X_mitra)))
    log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(mitra_time, 4)}s\n'

    log += '\nDash2002:\n'
    results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred3, ref, None, X_dash)))
    log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(dash_time, 4)}s\n\n'

    return log


if __name__ == '__main__':
    log_file = open('logs/evaluation_classic_methods.txt', 'a', newline='\n')

    SEED = 42
    nRep = 100
    
    datasets = [15]
    pVars = [0.50]

    for d in datasets:
        for p in pVars:
            log = evaluate(d, p, 1, nRep, SEED)
            print(log)
            #log_file.write(log)

    print(f"{'-'*30}> Done <{'-'*30}")
    log_file.close()