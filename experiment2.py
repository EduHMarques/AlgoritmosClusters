import numpy as np
import pandas as pd
import random

def experiment(dataset, mc, nRep, nClusters):
  nObj = len(dataset)
  print('nObj: ', nObj)
  print('nClusters: ', nClusters)

  centersAll = np.zeros((nRep, nClusters))

  for i in range(nRep):
    centersAll[i] = random.sample(range(1, nObj), nClusters)

  # print(centersAll)

  Jmin = 2147483647     # int_max do R
  partMin = 0

  for i in range(nRep):
    print('I: ', i)
    centers = list(map(int, centersAll[i,].tolist()))
    # print('Centros: ', centers)
    # print('Centros: ', centers)
    # print(type(centers[0]))
    resp = FCM(dataset, centers, 2)

    J = resp[0]
    L_resp = resp[1]
    M_resp = resp[2]

    if (Jmin > J):
      Jmin = J
      partMin = L_resp
    
  print('Jmin: ', J)
  print('Matriz: ', L_resp)

####################################################

def FCM(data, centers, parM):
 
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
  
  result = [J, L, Ubefore, count]
  return result
 
def initializePrototypes(data,centers):
 
  nVar = data.shape[1]
  nProt = len(centers)
  # print(f'Numero de var: {nVar}\nNumero de prot: {nProt}')
  
  P = np.arange(nProt * nVar)
  P = P.reshape((nProt, nVar))
  
  P = (np.zeros_like(P)).astype('float64')
  
  for k in range(0, nProt):
    P[k,] = data[centers[k],]
    # print(P[k,])

  # print(P)    # Não tá inicializando corretamente a matriz P.
  
  return P
 
def updatePrototypes(data, memberships, parM):
  
  nObj = data.shape[0]
  nVar = data.shape[1]
  nProt = memberships.shape[1]
  
  P = np.arange(nProt * nVar)
  P = P.reshape((nProt, nVar))
  P = (np.zeros_like(P)).astype('float64')
  
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
  
  D = np.arange(nObj * nProt)
  D = D.reshape((nObj, nProt))
  
  D = (np.zeros_like(D)).astype('float64')
  
  for j in range(0, nObj):
    
    obj = data[j,]
    
    for k in range(0, nProt):
        prot = prototypes[k,]
        
        # distance = sum((obj - prot)^2)
 
        distance = np.sum(np.square(np.subtract(obj, prot)))
        D[j,k] = distance
      
  return D
 
def updateMembership(distances, parM):
  
  nObj = distances.shape[0]
  nProt = distances.shape[1]
  
  U = np.arange(nObj * nProt)
  U = U.reshape((nObj, nProt))
  
  U = (np.zeros_like(U)).astype('float64')
  
  for i in range(0, nObj):
    
    for k in range(0, nProt):
      
      d = distances[i,k]
      soma = 0
      
      for kk in range(0, nProt):
        dd = distances[i,kk];       
        
        aux1 = ((int(d)+0.0000001)/(int(dd)+0.0000001))
        aux2 = (1.0/(parM-1.0))
        # soma += ((int(d)+0.0000001)/(int(dd)+0.0000001))^(1.0/(parM-1.0))
        soma += pow(aux1, aux2)
    
      soma = pow(soma, -1.0)
      
      U[i,k] = soma
  
  return U
 
def updateCriterion(memberships,distances,parM):
  
  J = 0
  
  nObj = distances.shape[0]
  nProt = distances.shape[1]
  
  for j in range(0, nObj):
    
    for k in range(0, nProt):
      
      delta = pow(memberships[j,k], parM)
      # print("Memberships:")
      # print(memberships[j,k])
      # print("Delta:")
      # print(delta)
      distance = distances[j,k]
      # print("Distancia:")
      # print(distance)
      
      J += delta*distance
  
  return J

def getPartition(memberships):
  
  L = []

  for object in memberships:
    L.append(np.argmax(object) + 1)
  
  return L

# Lendo data e iniciando o algoritmo:
 
# dataset = pd.read_csv("iris.txt", sep=",")
# dataset.columns = ['A', 'B', 'C', 'D', 'E']
 
# dataset_unlabeled = dataset.drop('E', axis=1)
# dataset_ref = dataset["E"].tolist()
# # print(dataset_ref)
 
# dataset_unlabeled = dataset_unlabeled.to_numpy()
# # print(dataset_unlabeled)

# nObj = len(dataset_unlabeled)
 
# centers = [135, 72, 110]

#############
 
# result = FCM(dataset_unlabeled, centers, 2)

# Jresult = result[0]
# membershipMatrix = result[1]

# correct_object = 0

# for i in range(len(membershipMatrix)):
#   if membershipMatrix[i] == dataset_ref[i]:
#     correct_object += 1

# print('Acertos: ', correct_object)
# print('Precisao: ', correct_object / nObj)

# experiment(dataset_unlabeled, 1, 100, 3)

 
# print(f"\nJ:", result[0])     # Critério
 
# print(f"\nL:", result[1])     # Lista de pertinencia de cada objeto
 
# # print("\nUbefore:")
# # print(result[2])
 
# print(f"\nCount: ", result[3])
 
# for item in result:
#   print(item)
#   print("\n\n")

