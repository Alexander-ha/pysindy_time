import sys
import numpy as np

class LassoTimeRegression:
    def __init__(self, learning_rate, iterations, l1_penalty, bandwidth, kernel):
        self.iterations = iterations
        self.l1_penalty = l1_penalty
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.tol = 1e-6

        self.all_W = None
        self.all_b = None
        self.t_values_ = None

    def gen_weighter(self, t_vals, H):
        def weighter(t):
            u = np.abs(t_vals - t) / H
            wcap = self.kernel(u)
            S = np.sum(wcap)
            if S == 0:
                return np.ones_like(t_vals) / len(t_vals)
            return H * wcap / S
        return weighter

    def _fit_one(self, X, Y, weights, lambda_eff):
        T, n = X.shape
        W = np.zeros(n)
        b = 0.0

        for epoch in range(self.iterations):
            pred = X @ W + b
            resid = Y - pred

            wres = np.sum(weights * resid)
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                b_new = wres / sum_weights +b
            else:
                b_new = 0.0

            if np.abs(b_new - b) > self.tol:
                b = b_new
                pred = X @ W + b
                resid = Y - pred

            max_change = 0.0
            for j in range(n):
                resid_partial = resid + W[j] * X[:, j]
                rho_j = np.sum(weights * X[:, j] * resid_partial)
                z_j = np.sum(weights * X[:, j] ** 2)
                if z_j < 1e-12:
                    continue

                # Soft-thresholding
                if rho_j > lambda_eff:
                    W_new = (rho_j - lambda_eff) / z_j
                elif rho_j < -lambda_eff:
                    W_new = (rho_j + lambda_eff) / z_j
                else:
                    W_new = 0.0

                change = np.abs(W_new - W[j])
                if change > max_change:
                    max_change = change

                if change > 0:
                    old_Wj = W[j]
                    W[j] = W_new
                    pred += (W_new - old_Wj) * X[:, j]
                    resid = Y - pred

            if max_change < self.tol:
                break

        return W, b

    def fit_all(self, X, Y, t_list=None):
        T, n = X.shape
        if t_list is None:
            t_list = np.arange(T)
        
        self.t_vals_ = np.asarray(t_list)
        self.t_values_ = self.t_vals_
        
        weighter = self.gen_weighter(self.t_vals_, self.bandwidth)
        weights_matrix = np.array([weighter(t) for t in t_list])
        lambda_eff = self.l1_penalty / self.bandwidth
        
        all_W = np.zeros((len(t_list), n))
        all_b = np.zeros(len(t_list))

        for idx, t in enumerate(t_list):
            weights = weights_matrix[idx]
            if np.sum(weights) == 0:
                all_W[idx] = 0
                all_b[idx] = 0
            else:
                W, b = self._fit_one(X, Y, weights, lambda_eff)
                all_W[idx] = W
                all_b[idx] = b

        self.all_W = all_W
        self.all_b = all_b
        return self

    def predict(self, t, X):
        X = np.asarray(X)   
        if self.all_W is None or self.all_b is None:
            raise RuntimeError("model wasn't fitted, call fit_all().")

        idx = np.argmin(np.abs(self.t_values_ - t))
        W = self.all_W[idx]
        b = self.all_b[idx]
        
        single = False
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single = True
        y_pred = X @ W + b
        return y_pred.item() if single else y_pred