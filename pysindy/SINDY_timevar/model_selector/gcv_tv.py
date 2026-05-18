from scipy import optimize
import numpy as np

class OutSampleCVSelector:
    def __init__(self, model, X, y, t=None, h_min=0.03, h_max=0.3, thresholdICI=1.0, subinterval=1, bootstrap=50, timemeanICI=False):
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
    
    def compute_std_local(self, time_moment, bandwidth):
        t_array = self.model.t_values_
        idx = np.argmin(np.abs(t_array - time_moment))
        
        half = max(5, int(bandwidth / (2 * (t_array[1] - t_array[0]))))
        start = max(0, idx - half)
        end = min(len(t_array), idx + half)
        
        coefs = []
        for i in range(start, end, max(1, (end-start)//10)):
            W = self.estimate_coefs(t_array[i], bandwidth)
            coefs.append(W)
        
        if len(coefs) < 3:
            return np.ones(self.n) * 0.01
        
        return np.std(coefs, axis=0)
    
    
    def _fit_without_point(self, idx_leave_out, bandwidth):
        """
        Fit model on all data except point idx_leave_out
        Returns coefficients estimated without that point
        """
        X_train = np.delete(self.X, idx_leave_out, axis=0)
        y_train = np.delete(self.y, idx_leave_out, axis=0)
        t_train = np.delete(self.model.t_values_, idx_leave_out, axis=0)
        
        original_bandwidth = self.model.bandwidth
        self.model.bandwidth = bandwidth
        
        t_left_out = self.model.t_values_[idx_leave_out]
        weights = self.model._compute_weights(t_left_out, t_train, bandwidth)
        
        if np.isscalar(bandwidth):
            lambda_eff = self.model.l1_penalty / bandwidth if bandwidth > 0 else self.model.l1_penalty
        else:
            idx = np.argmin(np.abs(self.model.t_values_ - t_left_out))
            lambda_eff = self.model.l1_penalty / bandwidth[idx] if bandwidth[idx] > 0 else self.model.l1_penalty
        
        try:
            W_cv, _ = self.model._fit_one_sklearn(X_train, y_train, weights, lambda_eff, time_idx=0)
            y_pred = X_train @ W_cv
            X_left_out = self.X[idx_leave_out:idx_leave_out+1, :]
            y_pred_left_out = X_left_out @ W_cv
            self.model.bandwidth = original_bandwidth
            return y_pred_left_out[0]
        except Exception:
            self.model.bandwidth = original_bandwidth
            return np.nan
    
    def compute_loocv_score(self, bandwidth):
        """
        Compute LOOCV score for a given bandwidth
        LOOCV = (1/n) * sum_{i=1}^n (y_i - y_hat_{-i})^2
        """
        if bandwidth < self.h_min or bandwidth > self.h_max:
            return np.inf
        
        squared_errors = []
        n_points = self.T
        
        print(f"    Computing LOOCV for h={bandwidth:.4f}...")
        
        for idx in range(n_points):
            y_pred_cv = self._fit_without_point(idx, bandwidth)
            if not np.isnan(y_pred_cv):
                err = (self.y[idx] - y_pred_cv) ** 2
                squared_errors.append(err)
        
        if len(squared_errors) == 0:
            return np.inf
        
        return np.mean(squared_errors)
    
    def compute_efficient_loocv(self, bandwidth):
        """
        Efficient LOOCV using hat matrix diagonal (if available)
        Assumes linear regression with weights
        Returns: LOOCV score
        """
        if bandwidth < self.h_min or bandwidth > self.h_max:
            return np.inf
        
        t_array = self.model.t_values_
        y_hat_all = np.zeros(self.T)
        leverage_sums = 0.0
        
        for idx in range(self.T):
            t = t_array[idx]
            
            original_bandwidth = self.model.bandwidth
            self.model.bandwidth = bandwidth
            weights = self.model._compute_weights(t, t_array, bandwidth)
            
            if np.isscalar(bandwidth):
                lambda_eff = self.model.l1_penalty / bandwidth if bandwidth > 0 else self.model.l1_penalty
            else:
                h_idx = np.argmin(np.abs(t_array - t))
                lambda_eff = self.model.l1_penalty / bandwidth[h_idx] if bandwidth[h_idx] > 0 else self.model.l1_penalty
            
            try:
                W_full, _ = self.model._fit_one_sklearn(self.X, self.y, weights, lambda_eff, time_idx=0)
                y_hat_all[idx] = self.X[idx] @ W_full
                
                leverage = weights[idx] / (np.sum(weights) + 1e-8)
                leverage_sums += leverage
                
            except Exception:
                y_hat_all[idx] = np.nan
                leverage = 0
            
            self.model.bandwidth = original_bandwidth
        
        valid_mask = ~np.isnan(y_hat_all)
        if np.sum(valid_mask) == 0:
            return np.inf
        
        loocv_errors = []
        for idx in range(self.T):
            if valid_mask[idx]:
                t = t_array[idx]
                self.model.bandwidth = bandwidth
                weights = self.model._compute_weights(t, t_array, bandwidth)
                H_ii = weights[idx] / (np.sum(weights) + 1e-8)
                residual = self.y[idx] - y_hat_all[idx]
                loocv_err = (residual / (1 - H_ii + 1e-8)) ** 2
                loocv_errors.append(loocv_err)
        
        return np.mean(loocv_errors)
    
    def optimize_loocv(self, h_grid=None, method='grid', efficient=True):
        """
        Optimize bandwidth using LOOCV
        
        Parameters:
        -----------
        h_grid : array, optional
            Grid of bandwidth values to search
        method : str
            'grid' for grid search, 'brent' for Brent optimization
        efficient : bool
            If True, use efficient hat-matrix LOOCV (faster)
            If False, use exact leave-one-out (slower but more accurate)
        
        Returns:
        --------
        best_h : float
            Optimal bandwidth
        best_score : float
            LOOCV score at optimal bandwidth
        """
        if method == 'grid':
            if h_grid is None:
                h_grid = np.linspace(self.h_min, self.h_max, 20)
            h_grid = h_grid[(h_grid >= self.h_min) & (h_grid <= self.h_max)]
            
            if len(h_grid) == 0:
                raise ValueError("no available bandwidth")
            
            scores = []
            print("  LOOCV grid search:")
            
            for h in h_grid:
                if efficient:
                    score = self.compute_efficient_loocv(h)
                else:
                    score = self.compute_loocv_score(h)
                scores.append(score)
                print(f"    h={h:.4f} -> LOOCV={score:.6f}")
            
            best_idx = np.argmin(scores)
            best_h = h_grid[best_idx]
            best_score = scores[best_idx]
            
            self.model.bandwidth = best_h
            return best_h, best_score
        
        elif method == 'brent':
            def objective(h):
                if efficient:
                    return self.compute_efficient_loocv(h)
                else:
                    return self.compute_loocv_score(h)
            
            res = optimize.minimize_scalar(
                objective,
                bounds=(self.h_min, self.h_max),
                method='bounded'
            )
            self.model.bandwidth = res.x
            return res.x, res.fun
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    
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
                    stds = self.compute_std_local(t, h)
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

    def optimize_all(self, h_grid=None, method='grid', cv_type='gcv'):
        """
        Optimize bandwidth using different methods
        
        Parameters:
        -----------
        h_grid : array, optional
            Grid of bandwidth values
        method : str
            'grid' or 'brent' for optimization method
        cv_type : str
            'gcv' - Generalized Cross-Validation (original)
            'loocv' - Leave-One-Out Cross-Validation (efficient)
            'loocv_exact' - Exact LOOCV (slower)
            'ICI' - ICI method (adaptive)
        
        Returns:
        --------
        best_h : float or array
            Optimal bandwidth(s)
        best_score : float
            Optimal score
        """
        if cv_type == 'gcv':
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
        
        elif cv_type == 'loocv':
            return self.optimize_loocv(h_grid, method, efficient=True)
        
        elif cv_type == 'loocv_exact':
            return self.optimize_loocv(h_grid, method, efficient=False)
        
        elif cv_type == 'ICI':
            if self.timemean_onreturn == True:
                opt_bands = self.select_pilot_bandwidthICI(threshold=self.thresholdICI)
                self.model.bandwidth = opt_bands
                return np.mean(opt_bands, axis=1), 0.0
            else:
                opt_bands = self.select_pilot_bandwidthICI(threshold=self.thresholdICI)
                self.model.bandwidth = opt_bands
                return opt_bands, 0.0
        
        else:
            raise ValueError(f"Unknown cv_type: {cv_type}. Use 'gcv', 'loocv', 'loocv_exact', or 'ICI'")