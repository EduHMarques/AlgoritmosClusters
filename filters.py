import numpy as np

# MFCM Based Filters

def sum_filter(data, U, nClusters):
    V = []
    nVar = data.shape[1]

    for i in range(0, nVar):
        aTotal = 0 # Relevância total
        nObj = data.shape[0]

        for j in range(0, nClusters):  
            a = 0
            for k in range(0, nObj):
                u = U[i][k][j] # Acessa o grau de pertinência do objeto k ao cluster j em relação a variável i
                a += u
        
            a /= nObj # Relevância em relação ao cluster j
            aTotal += a

        aTotal = round(aTotal * 100, 2)
        V.append(aTotal)
    
    return (V, 'Filtro por somatorio')


def variance_filter(data, U, nClusters):
    V = []
    nVar = data.shape[1]

    for i in range(0, nVar):
        # print(f'i: {i}')
        aTotal = [] # Relevância total
        nObj = data.shape[0]

        for j in range(0, nClusters):  
            a = 0
            for k in range(0, nObj):
                u = U[i][k][j] # Acessa o grau de pertinência do objeto k ao cluster j em relação a variável i
                a += u
        
            a /= nObj # Relevância em relação ao cluster j
            aTotal.append(a)

        aTotal = np.asarray(aTotal)
        # print(aTotal)
        variance = np.var(aTotal)

        V.append(round(variance * 100, 5))
    
    return (V, 'Filtro por variancia')


def variance_filter2(data, U, nClusters):
    V = []
    nVar = data.shape[1]

    for i in range(0, nVar):
        # print(f'i: {i}')
        aTotal = [] # Relevância total
        nObj = data.shape[0]

        for j in range(0, nClusters):
            a = []
            for k in range(0, nObj):
                u = U[i][k][j] # Acessa o grau de pertinência do objeto k ao cluster j em relação a variável i
                a.append(u)
        
            a = np.asarray(a)
            variancia = np.var(a)
            aTotal.append(variancia)

        aTotal = np.asarray(aTotal)
        # print(aTotal)
        media = np.mean(aTotal)

        V.append(round(media * 100, 5))
    
    return (V, 'Filtro por variancia 2')

