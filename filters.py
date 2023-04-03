import numpy as np

def sum_filter(data, U, nClusters):
    V = []
    nVar = data.shape[0]

    for i in range(0, nVar):
        aTotal = 0 # Relevância total
        nObj = data.shape[1]

        for j in range(0, nClusters):  
            a = 0
            for k in range(0, nObj):
                u = U[i][k][j] # Acessa o grau de pertinência do objeto k ao cluster j em relação a variável i
                a += u
        
            a /= nObj # Relevância em relação ao cluster j
            aTotal += a

        V.append(aTotal)
    
    return V


def variance_filter(data, U, nClusters):
    V = []
    nVar = data.shape[0]

    for i in range(0, nVar):
        aTotal = [] # Relevância total
        nObj = data.shape[1]

        for j in range(0, nClusters):  
            a = 0
            for k in range(0, nObj):
                u = U[i][k][j] # Acessa o grau de pertinência do objeto k ao cluster j em relação a variável i
                a += u
        
            a /= nObj # Relevância em relação ao cluster j
            aTotal.append(a)

        aTotal = np.asarray(aTotal)
        variance = np.var(aTotal)

        V.append(variance)
    
    return V