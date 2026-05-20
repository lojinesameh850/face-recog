import os
import glob
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Reuse the K-Means primitives so the GMM uses the exact same k-means initialisation
# and the exact same cluster-to-subject majority-vote mapping as the K-Means part.
from k_means_implementation import run_kmeans, map_clusters_to_subjects


class GaussianMixtureScratch:

    def __init__(self, n_components, max_iters=200, tol=1e-3, reg_covar=1e-6, seed=42):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed

    def _estimate_log_prob(self, X):
        # Log density of every sample under every component, using one shared
        # (tied) diagonal covariance -> (N, K).
        D = X.shape[1]
        precision = 1.0 / self.covariances_                       # (D,)
        log_det = np.sum(np.log(self.covariances_))               # scalar
        # (x - mu)^2 * precision, expanded so it stays a single vectorised expression.
        quad = ((X ** 2) @ precision)[:, np.newaxis] \
            - 2.0 * X @ (self.means_ * precision).T \
            + (self.means_ ** 2) @ precision                      # (N, K)
        return -0.5 * (D * np.log(2.0 * np.pi) + log_det + quad)

    def _e_step(self, X):
        # Weighted log probabilities, normalised with log-sum-exp for numerical stability.
        weighted_log_prob = self._estimate_log_prob(X) + np.log(self.weights_)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm.mean(), log_resp

    def _m_step(self, X, resp):
        # Update weights, means and one shared (tied) diagonal covariance.
        N = X.shape[0]
        nk = resp.sum(axis=0) + 1e-10                              # (K,)
        self.weights_ = nk / nk.sum()
        self.means_ = (resp.T @ X) / nk[:, np.newaxis]
        # Pool the per-component scatter into a single shared covariance vector.
        scatter = (resp.T @ (X ** 2)) - nk[:, np.newaxis] * self.means_ ** 2  # (K, D)
        self.covariances_ = scatter.sum(axis=0) / N + self.reg_covar          # (D,)

    def fit(self, X):
        np.random.seed(self.seed)
        N = X.shape[0]

        # Initialisation: K-Means hard assignment, then one M-step (same scheme as sklearn).
        centroids, kmeans_labels = run_kmeans(X, self.n_components)
        resp = np.zeros((N, self.n_components))
        resp[np.arange(N), kmeans_labels] = 1.0
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)
        self.means_ = centroids
        self._m_step(X, resp)

        # EM iterations: stop once the mean log-likelihood stops improving.
        prev_ll = -np.inf
        self.n_iter_ = self.max_iters
        for iteration in range(self.max_iters):
            ll, log_resp = self._e_step(X)
            self._m_step(X, np.exp(log_resp))
            if abs(ll - prev_ll) < self.tol:
                self.n_iter_ = iteration + 1
                break
            prev_ll = ll
        self.lower_bound_ = ll
        return self

    def predict(self, X):
        _, log_resp = self._e_step(X)
        return np.argmax(log_resp, axis=1)


class GaussianMixtureLibrary:
    def __init__(self, n_components, max_iters=200, tol=1e-3, reg_covar=1e-6, seed=42):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.reg_covar = reg_covar
        self.seed = seed

    def fit(self, X):
        self._model = GaussianMixture(
            n_components=self.n_components, covariance_type='tied',
            max_iter=self.max_iters, tol=self.tol, reg_covar=self.reg_covar,
            init_params='kmeans', n_init=1, random_state=self.seed).fit(X)
        # Expose the learned parameters under the same names as the scratch model.
        self.weights_ = self._model.weights_
        self.means_ = self._model.means_
        self.covariances_ = self._model.covariances_
        self.n_iter_ = self._model.n_iter_
        self.lower_bound_ = self._model.lower_bound_
        return self

    def predict(self, X):
        return self._model.predict(X)


def cluster_accuracy(gmm, X, y_true, K):
    """Fit-free helper: cluster X, map clusters to subjects, return predictions + accuracy."""
    cluster_labels = gmm.predict(X)
    label_map = map_clusters_to_subjects(cluster_labels, y_true, K)
    y_pred = np.array([label_map[c] for c in cluster_labels])
    return y_pred, label_map, accuracy_score(y_true, y_pred)


def evaluate_on_test(gmm, X_test, y_test, label_map):
    """Apply a trained GMM + its train-derived cluster->subject map to the test set."""
    cluster_labels = gmm.predict(X_test)
    y_pred = np.array([label_map[c] for c in cluster_labels])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    return acc, f1, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GMM (from-scratch EM vs. sklearn) on PCA or Autoencoder projections.")
    parser.add_argument('--mode', type=str, choices=['pca', 'ae'], default='pca')
    args = parser.parse_args()

    np.random.seed(42)

    if args.mode == 'pca':
        data_dir, train_prefix, test_prefix, config_label = \
            'pca_projected_data', 'train_pca_', 'test_pca_', 'Alpha'
    else:
        data_dir, train_prefix, test_prefix, config_label = \
            'auto_encoder_projected_data', 'train_ae_', 'test_ae_', 'AE Config'

    train_files = glob.glob(os.path.join(data_dir, f"{train_prefix}*.npy"))
    if not train_files:
        raise FileNotFoundError(f"No files found matching {data_dir}/{train_prefix}*.npy")

    configs = sorted(os.path.basename(f).replace(train_prefix, '').replace('.npy', '')
                     for f in train_files)
    Ks = [20, 40, 60]
    impls = ['scratch', 'library']

    y_train = np.load('processed_data/y_train.npy')
    y_test = np.load('processed_data/y_test.npy')

    # accuracy tables, one curve per implementation
    acc_vs_K = {impl: {c: [] for c in configs} for impl in impls}
    acc_vs_config = {impl: {K: [] for K in Ks} for impl in impls}
    acc_diffs = []
    best = {impl: {'acc': -1.0} for impl in impls}

    print(f"\n{'='*78}\nGMM clustering  |  mode={args.mode}\n{'='*78}")
    header = f"{config_label:>10} {'K':>4} | {'scratch acc':>12} {'library acc':>12} " \
             f"{'diff':>8} | {'scratch s':>10} {'library s':>10}"
    print(header)
    print('-' * len(header))

    for config in configs:
        # float64: the autoencoder projections are saved as float32, which makes the
        # tied full-covariance Cholesky factorisation fail for the library GMM.
        X_train = np.load(os.path.join(data_dir, f'{train_prefix}{config}.npy')).astype(np.float64)

        for K in Ks:
            # --- our own implementation -------------------------------------
            t0 = time.time()
            gmm_s = GaussianMixtureScratch(n_components=K, max_iters=200,
                                           tol=1e-3, reg_covar=1e-3, seed=42).fit(X_train)
            time_s = time.time() - t0
            _, map_s, acc_s = cluster_accuracy(gmm_s, X_train, y_train, K)

            # --- sklearn library implementation -----------------------------
            t0 = time.time()
            gmm_l = GaussianMixtureLibrary(n_components=K, max_iters=200,
                                           tol=1e-3, reg_covar=1e-3, seed=42).fit(X_train)
            time_l = time.time() - t0
            _, map_l, acc_l = cluster_accuracy(gmm_l, X_train, y_train, K)

            acc_vs_K['scratch'][config].append(acc_s)
            acc_vs_K['library'][config].append(acc_l)
            acc_vs_config['scratch'][K].append(acc_s)
            acc_vs_config['library'][K].append(acc_l)

            print(f"{config:>10} {K:>4} | {acc_s:>12.4f} {acc_l:>12.4f} "
                  f"{acc_s - acc_l:>+8.4f} | {time_s:>10.3f} {time_l:>10.3f}")

            acc_diffs.append(acc_s - acc_l)

            for impl, gmm, lbl_map, acc in [('scratch', gmm_s, map_s, acc_s),
                                            ('library', gmm_l, map_l, acc_l)]:
                if acc > best[impl]['acc']:
                    best[impl] = {'acc': acc, 'config': config, 'K': K,
                                  'gmm': gmm, 'label_map': lbl_map}

    # ---- relation analysis (alpha vs acc, K vs acc) --------------------------
    print(f"\n{config_label} vs. accuracy (averaged over K):")
    for impl in impls:
        for config in configs:
            vals = acc_vs_K[impl][config]
            print(f"  [{impl:>7}] {config_label} {config}: mean acc = {np.mean(vals):.4f}")
    print("K vs. accuracy (averaged over configs):")
    for impl in impls:
        for K in Ks:
            vals = acc_vs_config[impl][K]
            print(f"  [{impl:>7}] K {K}: mean acc = {np.mean(vals):.4f}")

    # ---- accuracy plots: scratch (top row) vs library (bottom row) -----------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for r, impl in enumerate(impls):
        ax = axes[r, 0]
        for config in configs:
            ax.plot(Ks, acc_vs_K[impl][config], marker='o', label=f'{config_label} = {config}')
        ax.set_title(f'Accuracy vs. K  ({impl} GMM)')
        ax.set_xlabel('Number of Components (K)')
        ax.set_ylabel('Accuracy')
        ax.set_xticks(Ks)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize='small')

        ax = axes[r, 1]
        for K in Ks:
            ax.plot(configs, acc_vs_config[impl][K], marker='s', label=f'K = {K}')
        ax.set_title(f'Accuracy vs. {config_label}  ({impl} GMM)')
        ax.set_xlabel(config_label)
        ax.set_ylabel('Accuracy')
        ax.grid(True, linestyle='--', alpha=0.7)
        if args.mode == 'ae':
            ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize='small')
    plt.tight_layout()
    plot_path = f'gmm_accuracy_plots_{args.mode}.png'
    plt.savefig(plot_path)
    print(f"Saved accuracy plots to '{plot_path}'.")

    # ---- evaluate the best model of each implementation on the TEST set ------
    print(f"\n{'='*78}\nTest-set evaluation of the best GMM models\n{'='*78}")
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    test_summary = {}

    for ax, impl in zip(axes, impls):
        b = best[impl]
        X_test = np.load(os.path.join(data_dir, f'{test_prefix}{b["config"]}.npy')).astype(np.float64)
        acc, f1, cm = evaluate_on_test(b['gmm'], X_test, y_test, b['label_map'])
        test_summary[impl] = {'config': b['config'], 'K': b['K'],
                              'train_acc': b['acc'], 'test_acc': acc, 'test_f1': f1}

        print(f"[{impl}]  best {config_label}={b['config']}, K={b['K']}")
        print(f"   Train Accuracy: {b['acc']:.4f}")
        print(f"   Test  Accuracy: {acc:.4f}")
        print(f"   Test  F1-Score: {f1:.4f}")

        sns.heatmap(cm, annot=False, cmap='Blues', ax=ax)
        ax.set_title(f'{impl} GMM  ({config_label}={b["config"]}, K={b["K"]})\n'
                     f'Test Acc={acc:.4f}, F1={f1:.4f}')
        ax.set_xlabel('Predicted Subject')
        ax.set_ylabel('True Subject')
    plt.tight_layout()
    cm_path = f'gmm_confusion_matrix_{args.mode}.png'
    plt.savefig(cm_path)
    print(f"\nSaved confusion matrices to '{cm_path}'.")

    # ---- final scratch vs. library comparison --------------------------------
    print(f"\n{'='*78}\nScratch vs. Library GMM\n{'='*78}")
    s, l = test_summary['scratch'], test_summary['library']
    print(f"  scratch : {config_label}={s['config']}, K={s['K']}  "
          f"-> test acc={s['test_acc']:.4f}, F1={s['test_f1']:.4f}")
    print(f"  library : {config_label}={l['config']}, K={l['K']}  "
          f"-> test acc={l['test_acc']:.4f}, F1={l['test_f1']:.4f}")
    mean_abs_diff = np.mean(np.abs(acc_diffs))
    print(f"  mean |train-acc difference| across all runs: {mean_abs_diff:.4f}")
