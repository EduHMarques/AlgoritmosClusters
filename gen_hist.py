import numpy as np
import matplotlib.pyplot as plt
import ast  # Para converter as strings em listas de números
from sklearn.cluster import KMeans as sklearnKMeans
from experiment import *
from filters import *
from datasets import selectDataset
from timeit import default_timer as timer

def evaluate(indexData, mc, nRep, seed, mf_m_line, mf_v_line):
    ## Plotagem do histograma
    num_bins = 10
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    plt.hist(mf_m_line, bins=num_bins, color='blue', alpha=0.7)
    plt.xlabel('Relevância')
    plt.ylabel('Frequência')
    plt.title(f'MF_M')

    plt.subplot(1, 2, 2)
    plt.hist(mf_v_line, bins=num_bins, color='green', alpha=0.7)
    plt.xlabel('Relevância')
    plt.ylabel('Frequência')
    plt.title(f'MF_V')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    SEED = 42
    nRep = 100
    mc = 1
    
    datasets = [21]

    # Abrir arquivos mf_m e mf_v e ler as linhas
    with open('logs/mf_m.txt', 'r') as file_m, open('logs/mf_v.txt', 'r') as file_v:
        mf_m_lines = file_m.readlines()  # Lê todas as linhas do arquivo mf_m
        mf_v_lines = file_v.readlines()  # Lê todas as linhas do arquivo mf_v
    
    # O loop deve rodar 4 vezes, já que os arquivos possuem 4 linhas cada
    for i in range(4):
        # Pegando a linha correspondente de cada arquivo e convertendo para lista de números
        mf_m_line = ast.literal_eval(mf_m_lines[i].strip())
        mf_v_line = ast.literal_eval(mf_v_lines[i].strip())
        
        # Imprimir as listas convertidas para garantir que são numéricas
        print(mf_m_line)
        print(mf_v_line)

        # Executando a função evaluate com as listas convertidas
        evaluate(datasets[0], mc, nRep, SEED, mf_m_line, mf_v_line)
