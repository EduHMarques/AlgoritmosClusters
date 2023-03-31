import numpy as np

def sum_filter(data, i, U, nClusters):
    aTotal = 0 # Relevância total
    nObj = data.shape[1]

    for j in range(0, nClusters):  
        a = 0
        for k in range(0, nObj):
            u = U[i][k][j] # Acessa o grau de pertinência do objeto k ao cluster j em relação a variável i
            a += u
    
        a /= nObj # Relevância em relação ao cluster j
        aTotal += a

    return aTotal