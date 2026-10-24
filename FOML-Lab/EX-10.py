
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Generate sample data
X, y = make_blobs(n_samples=100, centers=3, random_state=42)

# Apply EM algorithm using Gaussian Mixture Model
model = GaussianMixture(n_components=3, random_state=42)
model.fit(X)

# Predict clusters
labels = model.predict(X)

print("Number of clusters:", model.n_components)
print("Cluster labels:", np.unique(labels))
print("First 10 predicted labels:", labels[:10])
print("Model converged:", model.converged_)

# Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.title("EM Algorithm - Gaussian Mixture Model")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
