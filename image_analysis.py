import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from experiment import execute, run_filter

# num_subjects = 40
# images_per_subject = 10
# image_shape = (112, 92)

nRep = 100

image_shape = (8, 8)
num_clusters = 10
num_samples = 1500

num_pixels = image_shape[0] * image_shape[1]
cut_percentage = 0.5

def load_mnist():
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist.data.to_numpy()[:num_samples]
    return data

def load_data():
    data = load_digits(as_frame=True).data.to_numpy()[:num_samples]
    print(data.shape)
    return data

# image_folder = "datasets/att-faces/"

# def load_images(image_folder):
#     data = []
#     for subject in range(1, num_subjects + 1):
#         subject_folder = os.path.join(image_folder, f"s{subject}")
#         for img_num in range(1, images_per_subject + 1):
#             img_path = os.path.join(subject_folder, f"{img_num}.pgm")
#             img = Image.open(img_path).convert("L")
#             img_array = np.array(img).flatten()  # Flatten para criar um vetor unidimensional
#             data.append(img_array)
#     return np.array(data)

# dataset = load_images(image_folder)
dataset = load_data()
print("Formato do dataset carregado:", dataset.shape)  # Deve ser (400, 10304)

dataset_normalized = dataset / 255.0
# dataset_normalized = np.abs(dataset_normalized - 1)
np.set_printoptions(threshold=np.inf)
# print(dataset_normalized)

centersMC = np.random.randint(0, dataset_normalized.shape[0], size=(nRep, num_clusters))
mfcm_result = execute(nRep, dataset_normalized, centersMC, 0)
variables_to_cut = int(num_pixels * cut_percentage)
filtered_dataset = run_filter('mean_image', dataset_normalized, mfcm_result, variables_to_cut, numClusters=num_clusters)

def display_image(dataset, index, image_shape):
    img_array = dataset[index].reshape(image_shape)
    plt.imshow(img_array, cmap="gray")
    plt.axis("off")
    plt.show()

print("Visualizando uma imagem filtrada:")
display_image(filtered_dataset, 0, image_shape) 