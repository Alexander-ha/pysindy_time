from scipy import optimize
import numpy as np

class OutSampleCVSelector:
    def __init__(self, model, X, y, t=None, h_min=0.03, h_max=0.3, thresholdICI=1.0, subinterval=1, bootstrap = 50, timemeanICI=False):
        self.model = model
        self.X = X
        self.y = y
        self.T, self.n = X.shape
        self.timemean_onreturn = timemeanICI
        if np.isscalar(model.bandwidth):
            self.init_h = float(model.bandwidth)
        else:
            self.init_h = 10.0   # fallback

        self.h_min = h_min
        self.h_max = h_max
        self.bnd_vec = []
        self.thresholdICI = thresholdICI
        self.subinterval_size = subinterval
        self.n_bootstrap = bootstrap


    def _get_bandwidth_at(self, time_moment):
        if np.isscalar(self.model.bandwidth):
            return self.model.bandwidth
        else:
            idx = np.argmin(np.abs(self.model.t_values_ - time_moment))
            return self.model.bandwidth[idx]

    def _compute_weights(self, t, t_array):
        h_t = self._get_bandwidth_at(t)
        return self.model._compute_weights(t, t_array, h_t)

    def estimate_coefs(self, time_moment, bandwidth):
        original_bandwidth = self.model.bandwidth
        self.model.bandwidth = bandwidth
        weights = self.model._compute_weights(time_moment, self.model.t_values_, bandwidth)
        if np.isscalar(bandwidth):
            lambda_eff = self.model.l1_penalty / bandwidth if bandwidth > 0 else self.model.l1_penalty
        else:
            idx = np.argmin(np.abs(self.model.t_values_ - time_moment))
            lambda_eff = self.model.l1_penalty / bandwidth[idx] if bandwidth[idx] > 0 else self.model.l1_penalty
        W, b = self.model._fit_one_sklearn(self.X, self.y, weights, lambda_eff, time_idx=0)
        self.model.bandwidth = original_bandwidth
        return W

    def compute_std_bootstrap(self, time_moment, bandwidth, n_bootstrap=50):
        try:
            original_bandwidth = self.model.bandwidth
            self.model.bandwidth = bandwidth
            weights = self.model._compute_weights(time_moment, self.model.t_values_, bandwidth)
            self.model.bandwidth = original_bandwidth
            
            if np.isscalar(bandwidth):
                lambda_eff = self.model.l1_penalty / bandwidth if bandwidth > 0 else self.model.l1_penalty
            else:
                idx = np.argmin(np.abs(self.model.t_values_ - time_moment))
                lambda_eff = self.model.l1_penalty / bandwidth[idx] if bandwidth[idx] > 0 else self.model.l1_penalty
            
            W_hat, _ = self.model._fit_one_sklearn(self.X, self.y, weights, lambda_eff, time_idx=0)
            y_pred = self.X @ W_hat
            residuals = self.y - y_pred
            
            bootstrap_coefs = []
            
            for _ in range(n_bootstrap):
                wild_multiplier = np.random.choice([-1.0, 1.0], size=len(self.y))
                                
                y_boot = y_pred + residuals * wild_multiplier
                
                try:
                    W_boot, _ = self.model._fit_one_sklearn(
                        self.X, y_boot, weights, lambda_eff, time_idx=-1
                    )
                    bootstrap_coefs.append(W_boot)
                except Exception:
                    continue
            
            if len(bootstrap_coefs) < max(10, n_bootstrap // 5):
                return np.ones(self.n) * 0.05 * bandwidth
            
            bootstrap_coefs = np.array(bootstrap_coefs)
            std_est = np.std(bootstrap_coefs, axis=0)
            
            n_eff = max(1.0, np.sum(weights))
            correction_factor = (len(self.y) / n_eff) ** 0.75
            std_est *= correction_factor
            
            min_std = 1e-3 * np.mean(np.abs(W_hat)) if np.mean(np.abs(W_hat)) > 1e-10 else 1e-3
            std_est = np.maximum(std_est, min_std)
            
            return std_est
        
        except Exception as e:
            print(f"  Bootstrap err for t={time_moment:.3f}, h={bandwidth:.4f}: {e}")
            return np.ones(self.n) * 0.01 * bandwidth


    def select_pilot_bandwidthICI(self, step=1.2, base_band=0.05, size=20, threshold=1.0, n_bootstrap=200):
        h_grider = np.array([base_band * (step ** j) for j in range(size)])
        h_grider = h_grider[(h_grider >= self.h_min) & (h_grider <= self.h_max)]
        
        if len(h_grider) == 0:
            raise ValueError("no available bandwidth in grid for ICI")
        
        n_features = self.n
        n_times = len(self.model.t_values_)
        opt_bands = np.zeros((n_features, n_times))
        
        print(f"  ICI: {n_times} times, {n_features} feature, {len(h_grider)} bandwidths")
        print(f"  grid for bandwidth: {h_grider}")
        
        for t_idx, t in enumerate(self.model.t_values_):
            coefs_by_h = []
            stds_by_h = []
            
            for h_idx, h in enumerate(h_grider):
                try:
                    coefs = self.estimate_coefs(t, h)
                    stds = self.compute_std_bootstrap(t, h, n_bootstrap=n_bootstrap)
                    coefs_by_h.append(coefs)
                    stds_by_h.append(stds)
                except Exception as e:
                    coefs = np.zeros(n_features)
                    stds = np.ones(n_features) * 1e10
                    coefs_by_h.append(coefs)
                    stds_by_h.append(stds)
            
            for k in range(n_features):
                L_intersection = -np.inf
                U_intersection = np.inf
                best_h_idx = 0
                
                for h_idx in range(len(h_grider)):
                    coef = coefs_by_h[h_idx][k]
                    std = stds_by_h[h_idx][k]
                    
                    L_j = coef - threshold * std
                    U_j = coef + threshold * std
                    L_intersection = max(L_intersection, L_j)
                    U_intersection = min(U_intersection, U_j)
                    
                    if L_intersection > U_intersection:
                        break
                    best_h_idx = h_idx
                
                opt_bands[k, t_idx] = h_grider[best_h_idx]

        return opt_bands
    
    def _compute_single_pred_and_trace(self, idx, h):
        if h < self.h_min or h > self.h_max:
            return 0.0, 0.0
        t = self.model.t_values_[idx]
        original_bandwidth = self.model.bandwidth
        self.model.bandwidth = h
        weights = self.model._compute_weights(t, self.model.t_values_, h)
        y_hat_t = np.sum(weights * self.y)
        s_tt = weights[idx]
        self.model.bandwidth = original_bandwidth
        return y_hat_t, s_tt

    def _compute_global_gcv(self, h):
        if h < self.h_min or h > self.h_max:
            return np.inf
        y_hat = np.zeros(self.T)
        trace_S = 0.0
        for idx in range(self.T):
            y_hat[idx], s_tt = self._compute_single_pred_and_trace(idx, h)
            trace_S += s_tt
        rss = np.sum((self.y - y_hat) ** 2)
        df = self.T - trace_S
        if df <= 1e-8 or df > self.T:
            return np.inf
        return self.T * rss / (df ** 2)

    def optimize_all(self, h_grid=None, method='grid'):
        if method == 'grid':
            if h_grid is None:
                h_grid = np.linspace(self.h_min, self.h_max, 20)
            h_grid = h_grid[(h_grid >= self.h_min) & (h_grid <= self.h_max)]
            if len(h_grid) == 0:
                raise ValueError("no available bandwidth")
            scores = []
            print("  GCV grid search:")
            for h in h_grid:
                score = self._compute_global_gcv(h)
                scores.append(score)
                print(f"    h={h:.4f} -> GCV={score:.6f}")
            best_h = h_grid[np.argmin(scores)]
            self.model.bandwidth = best_h
            return best_h, min(scores)

        elif method == 'brent':
            res = optimize.minimize_scalar(
                lambda h: self._compute_global_gcv(h),
                bounds=(self.h_min, self.h_max),
                method='bounded'
            )
            self.model.bandwidth = res.x
            return res.x, res.fun

        elif method == 'ICI':
            if self.timemean_onreturn==True:
                opt_bands = self.select_pilot_bandwidthICI(threshold=self.thresholdICI)
                self.model.bandwidth = opt_bands
                return np.mean(opt_bands, axis=1), 0.0
            else:
                opt_bands = self.select_pilot_bandwidthICI(threshold=self.thresholdICI)
                self.model.bandwidth = opt_bands
                return opt_bands, 0.0

        else:
            raise ValueError(f"Unknown method: {method}")