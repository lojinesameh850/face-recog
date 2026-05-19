import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def run_kmeans(X, K, max_iters=100, tol=1e-4):
    np.random.seed(42)
    random_indices = np.random.choice(X.shape[0], K, replace=False)
    centroids = X[random_indices]
    
    for iteration in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                new_centroids[k] = X[np.random.choice(X.shape[0])]
                
        centroid_shift = np.linalg.norm(new_centroids - centroids, axis=1).sum()
        if centroid_shift < tol:
            break
            
        centroids = new_centroids
        
    return centroids, labels

def map_clusters_to_subjects(cluster_labels, y_true, K):
    label_map = {}
    for k in range(K):
        indices = np.where(cluster_labels == k)[0]
        if len(indices) > 0:
            true_labels_in_cluster = y_true[indices]
            unique_labels, counts = np.unique(true_labels_in_cluster, return_counts=True)
            label_map[k] = unique_labels[np.argmax(counts)]
        else:
            label_map[k] = y_true[np.random.choice(len(y_true))]
    return label_map

def predict_subjects(X, centroids, label_map):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    cluster_assignments = np.argmin(distances, axis=1)
    return np.array([label_map[k] for k in cluster_assignments])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run K-Means on PCA or Autoencoder projections.")
    parser.add_argument('--mode', type=str, choices=['pca', 'ae'], default='pca')
    args = parser.parse_args()

    if args.mode == 'pca':
        data_dir = 'pca_projected_data'
        train_prefix = 'train_pca_'
        test_prefix = 'test_pca_'
        config_label = 'Alpha'
    else:
        data_dir = 'auto_encoder_projected_data'
        train_prefix = 'train_ae_'
        test_prefix = 'test_ae_'
        config_label = 'AE Config'

    train_files = glob.glob(os.path.join(data_dir, f"{train_prefix}*.npy"))
    if not train_files:
        raise FileNotFoundError(f"No files found matching {data_dir}/{train_prefix}*.npy")

    configs = [os.path.basename(f).replace(train_prefix, '').replace('.npy', '') for f in train_files]
    configs.sort()
    
    Ks = [20, 40, 60]
    
    acc_vs_K = {config: [] for config in configs}
    acc_vs_config = {K: [] for K in Ks}
    
    best_acc = 0
    best_model = {}
    
    y_train = np.load('processed_data/y_train.npy')
    
    for config in configs:
        X_train = np.load(os.path.join(data_dir, f'{train_prefix}{config}.npy'))
        
        for K in Ks:
            centroids, cluster_labels = run_kmeans(X_train, K)
            label_map = map_clusters_to_subjects(cluster_labels, y_train, K)
            y_pred_train = predict_subjects(X_train, centroids, label_map)
            acc = accuracy_score(y_train, y_pred_train)
            
            acc_vs_K[config].append(acc)
            acc_vs_config[K].append(acc)
            
            print(f"{config_label}: {config}, K: {K} | Training Accuracy: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_model = {
                    'config': config, 
                    'K': K, 
                    'centroids': centroids, 
                    'label_map': label_map
                }

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    for config in configs:
        plt.plot(Ks, acc_vs_K[config], marker='o', label=f'{config_label} = {config}')
    plt.title('Classification Accuracy vs. K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Accuracy')
    plt.xticks(Ks)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if args.mode == 'ae':
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    else:
        plt.legend()

    plt.subplot(1, 2, 2)
    for K in Ks:
        plt.plot(configs, acc_vs_config[K], marker='s', label=f'K = {K}')
    plt.title(f'Classification Accuracy vs. {config_label}')
    plt.xlabel(config_label)
    plt.ylabel('Accuracy')
    
    rotation_angle = 45 if args.mode == 'ae' else 0
    plt.xticks(rotation=rotation_angle, ha='right' if rotation_angle else 'center')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plot_filename = f'kmeans_accuracy_plots_{args.mode}.png'
    plt.savefig(plot_filename)
    print(f"\nSaved accuracy plots to '{plot_filename}'.")

    print(f"Testing Best Model ({config_label}: {best_model['config']}, K: {best_model['K']})...")
    
    X_test = np.load(os.path.join(data_dir, f'{test_prefix}{best_model["config"]}.npy'))
    y_test = np.load('processed_data/y_test.npy')
    
    y_pred_test = predict_subjects(X_test, best_model['centroids'], best_model['label_map'])
    
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    cm = confusion_matrix(y_test, y_pred_test)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f'Confusion Matrix ({config_label}={best_model["config"]}, K={best_model["K"]})')
    plt.xlabel('Predicted Subject')
    plt.ylabel('True Subject')
    cm_filename = f'kmeans_confusion_matrix_{args.mode}.png'
    plt.savefig(cm_filename)
    print(f"Saved confusion matrix to '{cm_filename}'.")