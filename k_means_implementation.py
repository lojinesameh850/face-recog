import numpy as np

alphas = [0.8, 0.85, 0.9, 0.95]
Ks = [20, 40, 60]

# Load projected training data for each alpha
for alpha in alphas:
	X_train = np.load(f'pca_projected_data/train_pca_{alpha}.npy')
	y_train = np.load('processed_data/y_train.npy')

	# Implement K-means for this X_train and optionally vary K over Ks