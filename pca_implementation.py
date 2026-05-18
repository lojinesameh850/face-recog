import os
import numpy as np
import matplotlib.pyplot as plt

def compute_and_save_pca(D_train):
    print("Computing PCA...")
    # Mean centering
    mean_face = np.mean(D_train, axis=0)
    D_train_centered = D_train - mean_face
    
    # Compute covariance matrix (for 10304 dimensions, this yields a 10304x10304 matrix)
    cov_matrix = np.cov(D_train_centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    os.makedirs('pca_model', exist_ok=True)
    np.save('pca_model/mean_face.npy', mean_face)
    np.save('pca_model/eigenvalues.npy', eigenvalues)
    np.save('pca_model/eigenvectors.npy', eigenvectors)
    
    print("PCA computation complete. Parameters saved to 'pca_model/'.")
    return mean_face, eigenvalues, eigenvectors

def project_and_save_subspaces(D_train, D_test, mean_face, eigenvalues, eigenvectors):
    alphas = [0.8, 0.85, 0.9, 0.95]
    total_variance = np.sum(eigenvalues)
    
    D_train_centered = D_train - mean_face; D_test_centered = D_test - mean_face
    
    os.makedirs('pca_projected_data', exist_ok=True)
    
    for alpha in alphas:
        # Find how many principal components are needed to retain alpha variance
        explained_variance_ratio = np.cumsum(eigenvalues) / total_variance
        num_components = np.argmax(explained_variance_ratio >= alpha) + 1
        
        W = eigenvectors[:, :num_components]
        
        # Project the dataset into the reduced PCA subspace
        Z_train = np.dot(D_train_centered, W); Z_test = np.dot(D_test_centered, W)
        
        print(f"Alpha {alpha}: Retained {num_components} dimensions.")
        
        np.save(f'pca_projected_data/train_pca_{alpha}.npy', Z_train)
        np.save(f'pca_projected_data/test_pca_{alpha}.npy', Z_test)

def visualize_sample_faces(mean_face, eigenvectors, D_train, sample_idx=0):
    img_shape = (112, 92)
    alphas = [0.8, 0.85, 0.9, 0.95]
    total_variance = np.sum(np.load('pca_model/eigenvalues.npy'))
    eigenvalues = np.load('pca_model/eigenvalues.npy')

    os.makedirs('pca_model', exist_ok=True)

    sample_indices = [sample_idx, sample_idx + 1, sample_idx + 2]
    D_train_centered = D_train - mean_face

    plt.figure(figsize=(15, 5 * len(sample_indices)))

    for row, idx in enumerate(sample_indices):
        original_face = D_train[idx].reshape(img_shape)
        plt.subplot(len(sample_indices), len(alphas) + 1, row * (len(alphas) + 1) + 1)
        plt.imshow(original_face, cmap='gray')
        plt.title(f"Original Face {idx}")
        plt.axis('off')

        np.save(f'pca_model/original_face_{idx}.npy', D_train[idx])
        plt.imsave(f'pca_model/original_face_{idx}.png', original_face, cmap='gray')

        for i, alpha in enumerate(alphas):
            explained_variance_ratio = np.cumsum(eigenvalues) / total_variance
            num_components = np.argmax(explained_variance_ratio >= alpha) + 1
            W = eigenvectors[:, :num_components]

            Z = np.dot(D_train_centered[idx], W)
            reconstructed_face = np.dot(Z, W.T) + mean_face
            recon_img = reconstructed_face.reshape(img_shape)

            plt.subplot(len(sample_indices), len(alphas) + 1, row * (len(alphas) + 1) + i + 2)
            plt.imshow(recon_img, cmap='gray')
            plt.title(f"Alpha: {alpha}\n(Dim: {num_components})")
            plt.axis('off')

            alpha_pct = int(alpha * 100)
            np.save(f'pca_model/reconstructed_face_{idx}_alpha_{alpha_pct}.npy', reconstructed_face)
            plt.imsave(f'pca_model/reconstructed_face_{idx}_alpha_{alpha_pct}.png', recon_img, cmap='gray')

    plt.tight_layout()
    figure_path = 'pca_model/reconstructed_samples.png'
    plt.savefig(figure_path)
    plt.show()
    print(f"Saved reconstructed face images and arrays to 'pca_model/'.")

if __name__ == "__main__":
    # Load previously saved training and test splits
    D_train = np.load('processed_data/D_train.npy')
    D_test = np.load('processed_data/D_test.npy')
    
    # Compute PCA (or load if already computed to save time)
    if not os.path.exists('pca_model/eigenvalues.npy'):
        mean_face, evals, evecs = compute_and_save_pca(D_train)
    else:
        mean_face = np.load('pca_model/mean_face.npy')
        evals = np.load('pca_model/eigenvalues.npy')
        evecs = np.load('pca_model/eigenvectors.npy')
        print("Loaded saved PCA parameters.")
        
    project_and_save_subspaces(D_train, D_test, mean_face, evals, evecs)
    
    visualize_sample_faces(mean_face, evecs, D_train, sample_idx=5)