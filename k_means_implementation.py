import numpy as np
import os
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
    alphas = [0.8, 0.85, 0.9, 0.95]
    Ks = [20, 40, 60]
    
    acc_vs_K = {alpha: [] for alpha in alphas}
    acc_vs_alpha = {K: [] for K in Ks}
    
    best_acc = 0
    best_model = {}

    print("=== Phase 1: Section 6.a - Training & Evaluation ===")
    
    y_train = np.load('processed_data/y_train.npy')
    
    for alpha in alphas:
        X_train = np.load(f'pca_projected_data/train_pca_{alpha}.npy')
        
        for K in Ks:
            centroids, cluster_labels = run_kmeans(X_train, K)
            label_map = map_clusters_to_subjects(cluster_labels, y_train, K)
            y_pred_train = predict_subjects(X_train, centroids, label_map)
            acc = accuracy_score(y_train, y_pred_train)
            
            acc_vs_K[alpha].append(acc)
            acc_vs_alpha[K].append(acc)
            
            print(f"Alpha: {alpha}, K: {K} | Training Accuracy: {acc:.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_model = {
                    'alpha': alpha, 
                    'K': K, 
                    'centroids': centroids, 
                    'label_map': label_map
                }

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for alpha in alphas:
        plt.plot(Ks, acc_vs_K[alpha], marker='o', label=f'Alpha = {alpha}')
    plt.title('Classification Accuracy vs. K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Accuracy')
    plt.xticks(Ks)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.subplot(1, 2, 2)
    for K in Ks:
        plt.plot(alphas, acc_vs_alpha[K], marker='s', label=f'K = {K}')
    plt.title('Classification Accuracy vs. Alpha')
    plt.xlabel('Retained Variance Threshold (Alpha)')
    plt.ylabel('Accuracy')
    plt.xticks(alphas)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('kmeans_accuracy_plots.png')
    print("\nSaved accuracy plots to 'kmeans_accuracy_plots.png'.")

    print(f"\n=== Phase 2: Section 7 - Best Model Evaluation ===")
    print(f"Testing Best Model (Alpha: {best_model['alpha']}, K: {best_model['K']})...")
    
    X_test = np.load(f'pca_projected_data/test_pca_{best_model["alpha"]}.npy')
    y_test = np.load('processed_data/y_test.npy')
    
    y_pred_test = predict_subjects(X_test, best_model['centroids'], best_model['label_map'])
    
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    cm = confusion_matrix(y_test, y_pred_test)
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues')
    plt.title(f'Confusion Matrix (Alpha={best_model["alpha"]}, K={best_model["K"]})')
    plt.xlabel('Predicted Subject')
    plt.ylabel('True Subject')
    plt.savefig('kmeans_confusion_matrix.png')
    print("Saved confusion matrix to 'kmeans_confusion_matrix.png'.")