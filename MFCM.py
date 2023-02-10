import numpy as np

maxIteration = 50
J = 1000

def MFCM(data, centers, parM):
	
    count = 0

    P = initializePrototypes(data,centers)

    Ubefore = 0
    Jbefore = J + 1.0

    while (Jbefore - J) > 0.0001 and count < maxIteration:

        count += 1
        
        D = updateDistances(data,P)

        U = updateMembership(D,parM)

        P = updatePrototypes(data,U,parM)

        Jbefore = J
        J = updateCriterion(U,D,parM)

        Ubefore = U	

    M = np.arrange(len(centers) * data.shape[1])
    M = M.reshape((len(centers), data.shape[1]))

    M = (np.ones_like(M)).astype('float64')

    memb = aggregateMatrix(Ubefore,M)
    L = getPartition(memb)

    resp = [J, L, memb, count]

    return resp
	

def initializePrototypes(data,centers):
 
  nVar = data.shape[1]
  nProt = len(centers)
  
  P = np.arange(nProt * nVar)
  P = P.reshape((nProt, nVar))
  
  P = (np.zeros_like(P)).astype('float64')
  
  for k in range(0, nProt):
    P[k,] = data[centers[k],]
  
  return P


def updatePrototypes(data, memberships, parM):
  
  nObj = data.shape[0]
  nVar = data.shape[1]
  nProt = memberships.shape[1]
  
  P = np.arange(nProt * nVar)
  P = P.reshape((nProt, nVar))
  P = (np.zeros_like(P)).astype('float64')

  for i in range(0, nVar):
  
    for k in range(0, nProt):
        
        s = 0
        ss = 0
        
        for j in range(0, nObj):
            
            delta = pow(memberships[j, k], parM)
            obj = data[j,i]
            
            s += delta*obj
            ss += delta
        
        s = s/ss
        P[k,i] = s
  
  return P

def updateDistances(data, prototypes):
  
  nObj = data.shape[0]
  nProt = prototypes.shape[0]
  nVar = data.shape[1]
  
  D = []
  
  for i in range(0, nVar):
    
    Dvar = np.arange(nObj * nProt)
    Dvar = Dvar.reshape((nObj, nProt))
    
    Dvar = (np.zeros_like(Dvar)).astype('float64')
  
    for j in range(0, nObj):
        
        obj = data[j,]
        
        for k in range(0, nProt):
            prot = prototypes[k,]
    
            distance = np.sum(np.square(np.subtract(obj, prot)))
            Dvar[j,k] = distance

    D.append(Dvar)
      
  return D


def updateMembership(distances, parM):
  
  nObj = distances.shape[0]
  nProt = distances.shape[1]
  nVar = len(distances)
  
  U = []

  for v in range(0, nVar):

    Uvar = np.arange(nObj * nProt)
    Uvar = Uvar.reshape((nObj, nProt))
    
    Uvar = (np.zeros_like(Uvar)).astype('float64')
  
    for i in range(0, nObj):
        
        for k in range(0, nProt):
        
            d = distances[i,k]
            soma = 0
            
            for kk in range(0, nProt):
                dd = distances[i,kk];       
                
                aux1 = ((int(d)+0.0000001)/(int(dd)+0.0000001))
                aux2 = (1.0/(parM-1.0))

                soma += pow(aux1, aux2)
        
            soma = pow(soma, -1.0)
        
            Uvar[i,k] = soma
    
    U.append(Uvar)   
  
  return U


def updateCriterion(memberships,distances,parM):
  
  J = 0
  
  nObj = distances.shape[0]
  nProt = distances.shape[1]
  nVar = len(distances)

  for i in range(0, nVar):
  
    for j in range(0, nObj):
        
        for k in range(0, nProt):
        
            delta = pow(memberships[i][j,k], parM)
            distance = distances[i][j,k]
            
            J += delta*distance
    
  return J


def aggregateMatrix(memberships, M):
    
    nObj = memberships.shape[0]
    nProt = memberships.shape[1]
    nVar = len(memberships)

    memb = np.arange(nObj * nProt)
    memb = memb.reshape((nObj, nProt))
    
    memb = (np.zeros_like(memb)).astype('float64')	

    for j in range(0, nObj):
        soma0 = 0
        
        for k in range(0,nProt):
            soma = 0
            
            for i in range(0, nVar):
                soma += M[k,i]*memberships[i][j,k]
                soma0 += M[k,i]*memberships[i][j,k]
            
            memb[j,k] = soma
        
        for k in range(0,nProt):
            memb[j,k] = memb[j,k]/soma0

    return memb


def computeAij(memberships):
    
    nProt = memberships.shape[1]
    nVar = len(memberships)

    M = np.arrange(nProt * nVar)
    M = M.reshape((nProt, nVar))

    M = (np.ones_like(M)).astype('float64')

    for j in range(0, nProt):
        for k in range(0,nVar):
            M[j,k] = np.sum(memberships[k][:,j])
            soma = 0
            
            for kk in range(0, nVar):
                soma =  soma + sum(memberships[kk][:,j])
            
            M[j,k] = M[j,k]/soma

    return M


def getPartition(memberships):
  
  L = []

  for object in memberships:
    L.append(np.argmax(object) + 1)
  
  return L