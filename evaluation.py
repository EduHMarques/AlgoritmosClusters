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


    log += f'*{"-"*30}* {dataset_name} *{"-"*30}*\n\n'
    log += f'Seed: {seed}\n'
    log += f'Dataset: {dataset_name} | n_samples: {n_samples} | n_features: {n_features} | n_clusters: {nclusters}\n'
    log += 'Methods: MaxVar, LS, Dash2002, Mitra2002'

    log += '\n\nParameters:\n'
    
    log += '\nMFCM\n'
    log += f'MC: {mc} | nRep: {nRep}\n'

    log += '\nRESULTS:\n'

    for p in pVar:
        numVar = int(p * n_features)

        ## MFCM
        result, mfcm_time, centers = exec_mfcm(indexData, mc, nRep)

        start = timer()
        dataset_varfilter = run_filter(1, dataset, result, ref, numVar, nclusters)
        end = timer()
        filvar_time = round(end - start, 4)

        start = timer()
        dataset_sumfilter = run_filter(2, dataset, result, ref, numVar, nclusters)
        end = timer()
        filsum_time = round(end - start, 4)

        ## MaxVar

        start = timer()
        maxVar_features = maxVar(dataset, p)
        end = timer()
        maxvar_time = end - start

        ## Laplacian Score:

        start = timer()
        LS_features = laplacian_score(dataset, p)
        end = timer()
        ls_time = end - start

        ## Dash2002

        # threshold = 0.5 * n_features
        threshold = numVar
        start = timer()

        model = Dash2002(dataset, threshold)
        entropy = model.execute(dataset)
        dash_features = model.forward_selection()

        end = timer()
        dash_time = end - start

        # log += f"Entropy: {entropy}\n"
        # log += (f"Variáveis selecionadas: {dash_features}")

        ## Mitra2002

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

        og_Kmeans = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
        og_Kmeans.fit(dataset)
        y_pred_og = og_Kmeans.labels_
        
        KMeansresultVar = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
        KMeansresultVar.fit(dataset_varfilter)
        KMeansresultVar = KMeansresultVar.labels_

        KMeansresultSum = sklearnKMeans(n_clusters=K, random_state=seed, n_init=10)
        KMeansresultSum.fit(dataset_sumfilter)
        KMeansresultSum = KMeansresultSum.labels_

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

        log += f'\n{p*100}% OF FEATURES:\n\n'

        log += 'KMeans sem filtro:\n'
        results = list(map(lambda x: round(x, 8), calculate_accuracy(y_pred_og, ref, None, dataset)))
        log += (f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}\n')

        log += '\nFiltro por Variância:\n'
        results = list(map(lambda x: round(x, 8), calculate_accuracy(KMeansresultVar, ref, None, dataset_varfilter)))
        log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(mfcm_time, 4)}s (MFCM), {filvar_time}s\n'

        log += '\nFiltro por Somatório:\n'
        results = list(map(lambda x: round(x, 8), calculate_accuracy(KMeansresultSum, ref, None, dataset_sumfilter)))
        log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(mfcm_time, 4)}s (MFCM), {filsum_time}s\n'

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
        log += f'ARI: {results[0]}  NMI: {results[1]}%  Silhouette: {results[2]}%  DB: {results[3]}  Time: {round(dash_time, 4)}s\n'

    log += '\n'
    return log


if __name__ == '__main__':
    log_file = open('logs/evaluation_classic_methods.txt', 'a', newline='\n')

    SEED = 42
    nRep = 25
    
    datasets = [1, 1]
    pVars = [0.25, 0.50]

    for d in datasets:
        log = evaluate(d, pVars, 1, nRep, SEED)
        print(log + '\n')
        log_file.write(log)

    print(f"\n{'-'*30}> Done <{'-'*30}")
    log_file.close()