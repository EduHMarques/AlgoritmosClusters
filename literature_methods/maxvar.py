import numpy as np

def maxVar(dataset, feature_percentage=0.25):
    n_samples, n_features = dataset.shape
    n_selected_features = int(feature_percentage * n_features)
    features = []

    for it in range(n_features):
        variance = np.var(dataset[:, it])
        features.append([it, variance])

    features = sorted(features, key=lambda x: x[1], reverse=True)
    features = [f[0] for f in features]
    
    return features[0:n_selected_features]