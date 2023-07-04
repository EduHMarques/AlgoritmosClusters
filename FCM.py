import numpy as np
from timeit import default_timer as timer

def FCM(data, centers, parM):
  
  start = timer()
  
  maxIteration = 100
  J = 1000
 
  count = 0
  
  P = initializePrototypes(data,centers)
  
  Ubefore = 0
  Jbefore = J + 1.0
  
  while abs(Jbefore - J) > 0.000001 and count < maxIteration:
    
    count += 1
    
    D = updateDistances(data,P)
    # print("Matrix D: ")
    # print(D)
    # print("\n\n")
 
    U = updateMembership(D,parM)
    # print("Matrix U: ")
    # print(U)
    # print("\n\n")
 
    P = updatePrototypes(data,U,parM)
    # print("Matrix P: ")
    # print(P)
    # print("\n\n")
    
    Jbefore = J
    J = updateCriterion(U,D,parM)
    # print('J: ', J)
    
    Ubefore = U 
  
  L = getPartition(Ubefore)

  end = timer()
  
  result = [J, L, Ubefore, count, end - start]
  return result
 
def initializePrototypes(data,centers):
 
  nVar = data.shape[1]
  nProt = len(centers)
  
  P = np.zeros((nProt, nVar), dtype='float64')
  
  for k in range(0, nProt):
    P[k,] = data[centers[k],]
  
  return P
 
def updatePrototypes(data, memberships, parM):
  
  nObj, nVar = data.shape
  nProt = memberships.shape[1]
  
  P = np.zeros((nProt, nVar), dtype='float64')
  
  for k in range(0, nProt):
    
    s = 0
    ss = 0
    
    for j in range(0, nObj):
        
        delta = pow(memberships[j, k], parM)
        obj = data[j,]
        
        s += delta*obj
        ss += delta
    
    s = s/ss
    P[k,] = s
  
  return P
  
def updateDistances(data, prototypes):
  
  nObj = data.shape[0]
  nProt = prototypes.shape[0]

  D = np.zeros((nObj, nProt), dtype='float64')
  
  for j in range(0, nObj):
    
    obj = data[j,]
    
    for k in range(0, nProt):
        prot = prototypes[k,]
        distance = np.sum(np.square(np.subtract(obj, prot)))
        
        D[j,k] = distance
      
  return D
 
def updateMembership(distances, parM):
  
  nObj, nProt = distances.shape
  
  U = np.zeros((nObj, nProt), dtype='float64')
  
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
      
      U[i,k] = soma
  
  return U
 
def updateCriterion(memberships,distances,parM):
  
  J = 0
  
  nObj, nProt = distances.shape
  
  for j in range(0, nObj):
    
    for k in range(0, nProt):
      
      delta = pow(memberships[j,k], parM)
      distance = distances[j,k]
      
      J += delta*distance
  
  return J

def getPartition(memberships):
  
  L = []

  for object in memberships:
    L.append(np.argmax(object) + 1)
  
  return L