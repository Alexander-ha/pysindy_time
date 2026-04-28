import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Lasso

class LassoTimeRegression:
    def __init__(self, iterations=1000, l1_penalty=0.0, bandwidth=1.0, kernel=None,
                 initial_conditions=None, tau=10, use_prior=False, fit_intercept=True,
                 prior_indices=None, intervals=None):
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.bandwidth = bandwidth               
        self.kernel = kernel
        self.tol = 1e-4
        self.all_W = None
        self.all_b = None
        self.t_values_ = None
        self.initial_conditions = initial_conditions
        self.tau = tau
        self.use_prior = use_prior
        self.fit_intercept = fit_intercept
        self.prior_indices = prior_indices if prior_indices is not None else [0]
        self.intervals = intervals

    def _preprocess_bandwidth(self, t_list):
        if np.isscalar(self.bandwidth):
            bw = np.full(len(t_list), self.bandwidth, dtype=float)
        else:
            bw = np.asarray(self.bandwidth, dtype=float)
            if bw.ndim != 1 or len(bw) != len(t_list):
                raise ValueError("bandwidth must be scalar or 1d array with len T")
        if self.intervals is not None and self.intervals > 1:
            T = len(t_list)
            edges = np.linspace(0, T, self.intervals + 1).astype(int)
            bw_smoothed = np.empty(T)
            for i in range(self.intervals):
                idx = slice(edges[i], edges[i+1])
                bw_smoothed[idx] = np.mean(bw[idx])
            bw = bw_smoothed
        return bw

    def _compute_weights(self, t, t_array, h_t):
        u = np.abs(t_array - t) / h_t
        w = self.kernel(u)
        S = np.sum(w)
        if S < 1e-12:
            return np.ones(len(t_array)) / len(t_array)
        return w / S

    def _augment_with_prior(self, X, Y, weights, time_idx):
        if (not self.use_prior or self.initial_conditions is None 
            or self.tau <= 0 or time_idx not in self.prior_indices):
            return X, Y, weights

        n_features = X.shape[1]
        tau_sqrt = np.sqrt(self.tau)

        X_prior = np.eye(n_features) * tau_sqrt
        Y_prior = self.initial_conditions * tau_sqrt
        weights_prior = np.ones(n_features)

        X_aug = np.vstack([X, X_prior])
        Y_aug = np.concatenate([Y, Y_prior])
        weights_aug = np.concatenate([weights, weights_prior])
        return X_aug, Y_aug, weights_aug

    def _fit_one_sklearn(self, X, Y, weights, lambda_eff, time_idx):
        X_proc, Y_proc, w_proc = self._augment_with_prior(X, Y, weights, time_idx)
        n_samples = X_proc.shape[0]
        sw = np.sqrt(w_proc)
        Xw = X_proc * sw[:, np.newaxis]
        Yw = Y_proc * sw

        if self.l1_penalty <= 0:
            if Xw.shape[1] == 1 and not self.fit_intercept:
                x = Xw[:, 0]
                num = np.dot(x, Yw)
                den = np.dot(x, x)
                if den < 1e-12:
                    W_val = 0.0
                else:
                    W_val = num / den
                return np.array([W_val]), 0.0
            else:
                X_design = Xw
                if self.fit_intercept:
                    X_design = np.column_stack([Xw, np.ones(n_samples)])
                try:
                    coef = np.linalg.lstsq(X_design, Yw, rcond=None)[0]
                except np.linalg.LinAlgError:
                    reg = 1e-8 * np.eye(X_design.shape[1])
                    coef = np.linalg.solve(X_design.T @ X_design + reg, X_design.T @ Yw)
                if self.fit_intercept:
                    return coef[:-1], coef[-1]
                else:
                    return coef, 0.0

        alpha = lambda_eff / (2.0 * n_samples) if n_samples > 0 else lambda_eff
        model = Lasso(alpha=alpha, fit_intercept=self.fit_intercept,
                      max_iter=self.iterations, tol=self.tol, selection='cyclic')
        model.fit(Xw, Yw)
        if self.fit_intercept:
            return model.coef_, model.intercept_
        else:
            return model.coef_, 0.0

    def fit_all(self, X, Y, t_list=None):
        T = X.shape[0]
        if t_list is None:
            t_list = np.arange(T)
        self.t_values_ = np.asarray(t_list)
        bw_array = self._preprocess_bandwidth(self.t_values_)
        if np.isscalar(self.bandwidth):
            lambda_eff = self.l1_penalty / self.bandwidth if self.bandwidth > 0 else self.l1_penalty
            lambda_eff_vec = np.full(len(t_list), lambda_eff)
        else:
            lambda_eff_vec = self.l1_penalty / np.maximum(bw_array, 1e-12) 

        n_features = X.shape[1]
        self.all_W = np.zeros((len(t_list), n_features))
        self.all_b = np.zeros(len(t_list))

        for idx, t in enumerate(t_list):
            h_t = bw_array[idx]
            weights = self._compute_weights(t, self.t_values_, h_t)
            W, b = self._fit_one_sklearn(X, Y, weights, lambda_eff_vec[idx], time_idx=idx)
            self.all_W[idx] = W
            self.all_b[idx] = b
        return self

    def projected_smoothing(self, data, sigma, constraints, n_iterations=10):
        smoothed = data.copy()
        for _ in range(n_iterations):
            if data.ndim == 1:
                smoothed = gaussian_filter1d(smoothed, sigma=sigma/2)
            else:
                for j in range(data.shape[1]):
                    smoothed[:, j] = gaussian_filter1d(smoothed[:, j], sigma=sigma/2)
            for idx, value in constraints.items():
                smoothed[idx] = value
        return smoothed

    def smooth_coefs(self):
        constraints_W = {0: self.all_W[0]}
        constraints_b = {0: self.all_b[0]}
        sigma = np.mean(self.bandwidth) / 5
        self.all_W = self.projected_smoothing(self.all_W, sigma=sigma, constraints=constraints_W)
        self.all_b = self.projected_smoothing(self.all_b, sigma=sigma, constraints=constraints_b)

    def predict(self, t, X):
        if self.all_W is None:
            raise RuntimeError("Model not fitted")
        idx = np.argmin(np.abs(self.t_values_ - t))
        W = self.all_W[idx]
        b = self.all_b[idx]
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return (X @ W + b).item() if X.ndim == 1 else X @ W + b