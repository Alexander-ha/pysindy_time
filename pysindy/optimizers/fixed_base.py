import copy
import numpy as np
from typing import Optional, Tuple, List, Union
from pysindy.optimizers.base import BaseOptimizer
from pysindy.utils import AxesArray, drop_nan_samples
from sklearn.utils.validation import check_X_y
from sklearn.linear_model import LinearRegression
from pysindy.optimizers.base import _preprocess_data, _normalize_features
from pysindy.SINDY_timevar.regressors.time_regressor import LassoTimeRegression
from pysindy.SINDY_timevar.model_selector.gcv_tv import OutSampleCVSelector
from types import SimpleNamespace
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline



class FixedCoefficientOptimizer:
    """
    Proxy-class for base SINDY optimizer that allows fixing coefficients with help of a binary mask.
    
    Params
    ----------
    base_optimizer : BaseOptimizer
        Basic optimizer for basic coefs problems
    fixed_coefs : Optional[np.ndarray]
        binary mask where True=fix(1) coefficient, False=optimize(0). Shape must match (n_targets : n_features)
    fixed_values : Optional[np.ndarray]
        fixed values for fixed coefficients that we don't want to adjust.
    time_varying_coefs : Optional[np.ndarray]
        binary mask for time‑varying coefficients.
    tv_optimizer_params : Optional[dict]
        parameters for LassoTimeRegression (if tv_optimizer not given).
    tv_optimizer : Optional[LassoTimeRegression]
        pre‑configured TV optimizer.
    no_normalization_for_fixeds : Optional[bool]
        if True, fixed coefficients are not denormalized (only relevant if normalize_columns=True)
    model_selector : Optional[OutSampleCVSelector] 
        selects models for different bandwidth parameter in time-varying case
    init_conds : Optional[np.ndarray]
        Initial conditions on time-varying coefficients to start the system
    options : Optional[dict] 
        could be broadcaster in following way:
        - use_selector : [bool] flag for selection
        - use_for_each : [bool] if use_selector is True then uses adaptive bandwidth selection.
        - selector_method: 'grid', 'newton', 'bfgs', 'ICI' - different methods to find bandwidth.
        [Warning!] use_for_each shows better perfomance only in very rare cases(extremelly varying coefficients)
        in current implementation, except the ICI method, it automatically applies adaptive bandwidth. Should be reviewed and modified. Recommendation: use default method for global bandwidth.
    """
    def __init__(self, base_optimizer: BaseOptimizer,
                 fixed_coefs: Optional[np.ndarray] = None,
                 fixed_values: Optional[np.ndarray] = None,
                 time_varying_coefs: Optional[np.ndarray] = None,
                 tv_optimizer_params: Optional[dict] = None,
                 tv_optimizer: Optional[LassoTimeRegression] = None,
                 no_normalization_for_fixeds: Optional[bool] = True,
                 model_selector: Optional[OutSampleCVSelector] = None,
                 init_conds: Optional[np.ndarray] = None,
                 options: Optional[dict] = None,
                 procopts: Optional[dict] = None, 
                 noise_level = 0.0, auto_preprocess = False):
        
        self.base_optimizer = copy.deepcopy(base_optimizer)
        self.fixed_coefs = fixed_coefs
        self.fixed_values = fixed_values
        self.time_varying_coefs = time_varying_coefs
        self.no_normalization_for_fixeds = no_normalization_for_fixeds
        self._is_fitted = False
        self.max_iter = 100
        self.model_selector = model_selector
        self.base_optimizer_class = base_optimizer.__class__
        self.base_optimizer_params = base_optimizer.get_params()
        self.noise_level = noise_level

        self.init_conds = init_conds
        default_options = {'selector_method': 'grid', 'use_selector': False, 
                           'use_for_each': False, 'smooth_coefs': False, 'use_time_meanICI': False, 'hmin': 0.3,
                             'hmax': 2.0, 'thresholdICI': 2.5, 'subinterval': 1, 'bootstrap': 30}
        preprocessor_defoptions = {'time_end': 1.0, 'time_start': 0.0, 'step': 0.1, 
                           's': 0, 'X': None}
        self.preprocessor_options = SimpleNamespace(**(preprocessor_defoptions | (procopts or {})))
        self.options = SimpleNamespace(**(default_options | (options or {})))
        if tv_optimizer is not None:
            self.tv_optimizer = tv_optimizer
        else:
            tv_optimizer_params = tv_optimizer_params or {}
            self.tv_optimizer = LassoTimeRegression(**tv_optimizer_params)
        self.auto_preprocess=auto_preprocess

    def auto_preprocessor(self):
        X_tv = self.preprocessor_options.X
        noise_std = np.std(self.preprocessor_options.X)
        x_smooth=np.zeros_like(X_tv); x_dot_smooth=np.zeros_like(X_tv)
        t=np.arange(self.preprocessor_options.time_start, self.preprocessor_options.time_end, self.preprocessor_options.step)
        for j in range(X_tv.shape[1]):
            spl=UnivariateSpline(t,X_tv[:,j],s=self.preprocessor_options.s)
            x_smooth[:,j]=spl(t); x_dot_smooth[:,j]=spl.derivative()(t)
        return x_dot_smooth

    def _init_fix_mask(self, coef_shape: Tuple[int, int]):
        """Initialization of both fixed and time‑varying masks."""
        if self.fixed_coefs is None:
            self.fixed_mask_ = np.zeros(coef_shape, dtype=bool)
            self.fixed_values_ = None
        else:
            fixed_coefs = np.asarray(self.fixed_coefs)
            if fixed_coefs.ndim == 1:
                if coef_shape[0] == 1:
                    fixed_coefs = fixed_coefs.reshape(1, -1)
                else:
                    raise ValueError(
                        f"fixed_coefs is 1d with shape {fixed_coefs.shape}, "
                        f"expected 2d {coef_shape} for {coef_shape[0]} outputs")
            self.fixed_mask_ = fixed_coefs.astype(bool)
            if self.fixed_mask_.shape != coef_shape:
                raise ValueError(
                    f"Shape of fixed_coefs {self.fixed_mask_.shape} "
                    f"does not match expected shape {coef_shape}")
            if self.fixed_values is not None:
                fixed_vals = np.asarray(self.fixed_values)
                if fixed_vals.ndim == 1 and coef_shape[0] == 1:
                    fixed_vals = fixed_vals.reshape(1, -1)
                self.fixed_values_ = np.where(self.fixed_mask_, fixed_vals, 0)
            else:
                self.fixed_values_ = None

        if self.time_varying_coefs is None:
            self.tv_mask_ = np.zeros(coef_shape, dtype=bool)
        else:
            tv_coefs = np.asarray(self.time_varying_coefs)
            if tv_coefs.ndim == 1 and coef_shape[0] == 1:
                tv_coefs = tv_coefs.reshape(1, -1)
            self.tv_mask_ = tv_coefs.astype(bool)
            if self.tv_mask_.shape != coef_shape:
                raise ValueError(
                    f"Shape of time_varying_coefs {self.tv_mask_.shape} "
                    f"does not match expected shape {coef_shape}")

        if np.any(self.fixed_mask_ & self.tv_mask_):
            raise ValueError("Both types (time-varying and constant) are specified for a coefficient. Please choose one.")
        
        if hasattr(self.tv_optimizer, 'bandwidth'):
            self.original_bandwidth = self.tv_optimizer.bandwidth

    def _get_bandwidth_for_coefficient(self, bandwidth_input, target_idx: int, coef_idx: int = None):
        """
        Extract bandwidth for a specific coefficient from various input formats.
        
        Parameters
        ----------
        bandwidth_input : scalar, list, or array
            Input bandwidth specification
        target_idx : int
            Index of the target (output dimension)
        coef_idx : int or None
            Index of the coefficient within the target. If None, returns bandwidth for entire target.
            
        Returns
        -------
        Bandwidth value for the specified coefficient, or None if not applicable.
        """
        if bandwidth_input is None:
            return None
            
        if np.isscalar(bandwidth_input):
            return bandwidth_input
            
        bandwidth_input = np.asarray(bandwidth_input)
        
        if bandwidth_input.ndim == 1:
            if target_idx < len(bandwidth_input):
                bw_for_target = bandwidth_input[target_idx]
                if isinstance(bw_for_target, (list, np.ndarray)):
                    if coef_idx is not None and coef_idx < len(bw_for_target):
                        return bw_for_target[coef_idx]
                    elif coef_idx is None:
                        return bw_for_target
                else:
                    return bw_for_target
            else:
                return None

        elif bandwidth_input.ndim == 2:
            if target_idx < bandwidth_input.shape[0]:
                if coef_idx is not None and coef_idx < bandwidth_input.shape[1]:
                    return bandwidth_input[target_idx, coef_idx]
                elif coef_idx is None:
                    return bandwidth_input[target_idx, :]
            return None
            
        return bandwidth_input

    def fit(self, x_, y, t=None, sample_weight=None, **reduce_kws):
        """
        Fit with fixed and time‑varying coefficients.
        """
        x_arr = np.asarray(x_)
        y_arr = np.asarray(y)
        
        if x_arr.ndim == 2:
            x_axes = {"ax_sample": 0, "ax_coord": 1}
        elif x_arr.ndim == 3:
            x_axes = {"ax_sample": 0, "ax_coord": 1, "ax_trajectory": 2}
        else:
            raise ValueError(f"x_ must be 2D or 3D, got {x_arr.ndim}D")
        x_ = AxesArray(x_arr, x_axes)

        if y_arr.ndim == 1:
            y_axes = {"ax_sample": 0}
        elif y_arr.ndim == 2:
            y_axes = {"ax_sample": 0, "ax_coord": 1}
        elif y_arr.ndim == 3:
            y_axes = {"ax_sample": 0, "ax_coord": 1, "ax_trajectory": 2}
        else:
            raise ValueError(f"y must be 1D, 2D or 3D, got {y_arr.ndim}D")
        y = AxesArray(y_arr, y_axes)

        x_, y = drop_nan_samples(x_, y)
        if self.auto_preprocess:
            y = self.auto_preprocessor()
        x_, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=True)
        x, y, X_offset, y_offset, _, sample_weight_sqrt = _preprocess_data(
            x_, y, fit_intercept=False, copy=self.base_optimizer.copy_X,
            sample_weight=sample_weight)
        if t is None:
            t = np.arange(x.shape[0])
        self.t_ = np.asarray(t)

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        coef_shape = (y.shape[1], x.shape[1])
        self._init_fix_mask(coef_shape)

        x_normed = np.copy(x)
        if self.base_optimizer.normalize_columns:
            feat_norms, x_normed = _normalize_features(x_normed)
            self.feat_norms_ = feat_norms

        # Initial coefficients
        if self.base_optimizer.initial_guess is None:
            initial_coef = np.linalg.lstsq(x_normed, y, rcond=None)[0].T
        else:
            initial_coef = self.base_optimizer.initial_guess
        self.coef_ = initial_coef.copy()
        if self.fixed_values_ is not None:
            self.coef_[self.fixed_mask_] = self.fixed_values_[self.fixed_mask_]

        n_targets = y.shape[1]
        self.reduced_indices_ = [None] * n_targets
        self.tv_models_ = [None] * n_targets
        self.tv_coefs_ = [None] * n_targets
        self.tv_biases_ = [None] * n_targets
        h_arr = []
        self.bandwidth_info_ = [None] * n_targets

        # First, fit initial constant coefficients to get residuals for ICI
        if self.options.use_selector:
            print("Computing initial residuals for bandwidth selection...")
            
            # Initial fit of constant coefficients (ignoring time-varying for now)
            const_coefs_initial = np.zeros((n_targets, x_normed.shape[1]))
            for k in range(n_targets):
                fixed_idx = self.fixed_mask_[k]
                tv_idx = self.tv_mask_[k]
                const_idx = ~(fixed_idx | tv_idx)
                
                if np.any(const_idx):
                    x_const = x_normed[:, const_idx]
                    y_const = y[:, k].copy()
                    
                    # Remove fixed contributions if any
                    if np.any(fixed_idx):
                        y_const -= x_normed[:, fixed_idx] @ self.coef_[k, fixed_idx]
                    
                    # Fit constant coefficients
                    opt_k = self.base_optimizer_class(**self.base_optimizer_params)
                    opt_k.fit(x_const, y_const, sample_weight=sample_weight, **reduce_kws)
                    const_coefs_initial[k, const_idx] = opt_k.coef_.flatten()
            
            # Now compute residuals and run ICI selector for each target
            for k in range(n_targets):
                tv_idx = self.tv_mask_[k]
                if not np.any(tv_idx):
                    h_arr.append(None)
                    self.bandwidth_info_[k] = None
                    continue

                # Compute residual after removing fixed and constant contributions
                y_residual = y[:, k].copy()
                
                # Remove fixed contributions
                fixed_idx = self.fixed_mask_[k]
                if np.any(fixed_idx):
                    y_residual -= x_normed[:, fixed_idx] @ self.coef_[k, fixed_idx]
                
                # Remove constant contributions
                const_idx = ~(self.fixed_mask_[k] | tv_idx)
                if np.any(const_idx):
                    y_residual -= x_normed[:, const_idx] @ const_coefs_initial[k, const_idx]
                
                # Create temporary TV optimizer for bandwidth selection
                temp_tv = copy.deepcopy(self.tv_optimizer)
                
                # Handle hmax/hmin as lists
                if isinstance(self.options.hmax, (list, np.ndarray)):
                    if k < len(self.options.hmax):
                        hmax_k = self.options.hmax[k]
                    else:
                        hmax_k = self.options.hmax[-1] if len(self.options.hmax) > 0 else 2.0
                else:
                    hmax_k = self.options.hmax
                    
                if isinstance(self.options.hmin, (list, np.ndarray)):
                    if k < len(self.options.hmin):
                        hmin_k = self.options.hmin[k]
                    else:
                        hmin_k = self.options.hmin[-1] if len(self.options.hmin) > 0 else 0.3
                else:
                    hmin_k = self.options.hmin

                # Select bandwidth using residuals
                selector = OutSampleCVSelector(
                    temp_tv, x_normed[:, tv_idx], y_residual, t=self.t_,
                    h_min=hmin_k,
                    h_max=hmax_k,
                    thresholdICI=self.options.thresholdICI, 
                    subinterval=self.options.subinterval,
                    timemeanICI=self.options.use_time_meanICI, 
                    bootstrap=self.options.bootstrap
                )
                best_h, _ = selector.optimize_all(method=self.options.selector_method)
                h_arr.append(best_h)
                self.bandwidth_info_[k] = best_h
                
                if isinstance(best_h, np.ndarray):
                    print(f"Target {k}: selected variable bandwidth (mean={np.mean(best_h):.4f})")
                else:
                    print(f"Target {k}: selected bandwidth = {best_h:.4f}")
                
                # Update initial constant coefficients with selected bandwidth info
                # Re-fit constant coefficients with updated bandwidth information
                if np.any(const_idx):
                    x_const = x_normed[:, const_idx]
                    y_const = y[:, k].copy()
                    if np.any(fixed_idx):
                        y_const -= x_normed[:, fixed_idx] @ self.coef_[k, fixed_idx]
                    
                    # Use the selected bandwidth for TV part if needed
                    opt_k = self.base_optimizer_class(**self.base_optimizer_params)
                    opt_k.fit(x_const, y_const, sample_weight=sample_weight, **reduce_kws)
                    self.coef_[k, const_idx] = opt_k.coef_.flatten()
        else:
            # If no selector, initialize h_arr with None
            for k in range(n_targets):
                h_arr.append(None)

        self.t_models = [None] * n_targets
        self._prev_tv_coefs = [None] * n_targets
        
        # Main optimization loop
        for iteration in range(self.max_iter):
            max_change = 0.0

            # Update constant coefficients
            for k in range(n_targets):
                fixed_idx = self.fixed_mask_[k]
                tv_idx = self.tv_mask_[k]
                const_idx = ~(fixed_idx | tv_idx)
                if not np.any(const_idx):
                    continue

                tv_pred = np.zeros(x_normed.shape[0])
                if np.any(tv_idx) and self.tv_models_[k] is not None:
                    tv_cols = np.where(tv_idx)[0]
                    for j, col in enumerate(tv_cols):
                        model_j = self.tv_models_[k][j]
                        for i, ti in enumerate(self.t_):
                            tv_pred[i] += model_j.predict(ti, x_normed[i, [col]].reshape(1, -1)).item()

                y_const = y[:, k].copy()
                if np.any(fixed_idx):
                    y_const -= x_normed[:, fixed_idx] @ self.coef_[k, fixed_idx]
                y_const -= tv_pred

                x_const = x_normed[:, const_idx]
                opt_k = self.base_optimizer_class(**self.base_optimizer_params)
                opt_k.fit(x_const, y_const, sample_weight=sample_weight, **reduce_kws)
                new_coef_const = opt_k.coef_.flatten()
                change = np.max(np.abs(new_coef_const - self.coef_[k, const_idx]))
                max_change = max(max_change, change)
                self.coef_[k, const_idx] = new_coef_const

            # Update time-varying coefficients
            for k in range(n_targets):
                tv_idx = self.tv_mask_[k]
                if not np.any(tv_idx):
                    continue

                fixed_idx = self.fixed_mask_[k]
                const_idx = ~(fixed_idx | tv_idx)

                const_pred = np.zeros(x_normed.shape[0])
                if np.any(fixed_idx):
                    const_pred += x_normed[:, fixed_idx] @ self.coef_[k, fixed_idx]
                if np.any(const_idx):
                    const_pred += x_normed[:, const_idx] @ self.coef_[k, const_idx]
                y_tv = y[:, k] - const_pred

                tv_cols = np.where(tv_idx)[0]
                n_tv = len(tv_cols)
                T = len(self.t_)

                if self.options.use_selector and k < len(h_arr) and h_arr[k] is not None:
                    bw_info = h_arr[k] 
                else:
                    bw_info = self.bandwidth_info_[k]
                    if bw_info is None and hasattr(self, 'original_bandwidth'):
                        bw_info = self.original_bandwidth

                if self.tv_coefs_[k] is not None and self.tv_coefs_[k].shape == (T, n_tv):
                    tv_coefs_k = self.tv_coefs_[k].copy()
                else:
                    tv_coefs_k = np.zeros((T, n_tv))
                tv_biases_k = np.zeros(T)
                print(f"Iter {iteration}, eq {k}: backfitting")
                max_backfit_iters = 30
                backfit_tol = 1e-6
                
                for backfit_iter in range(max_backfit_iters):
                    max_coef_change = 0.0
                    for j, col in enumerate(tv_cols):
                        y_residual = y_tv.copy()
                        for other_j in range(n_tv):
                            if other_j == j:
                                continue
                            y_residual -= x_normed[:, tv_cols[other_j]] * tv_coefs_k[:, other_j]

                        # Get bandwidth for this specific coefficient
                        bw_j = self._get_bandwidth_for_coefficient(bw_info, k, j) if bw_info is not None else None

                        tv_j = copy.deepcopy(self.tv_optimizer)
                        if self.init_conds is not None:
                            tv_j.initial_conditions = np.atleast_1d(self.init_conds[k, tv_idx][j])
                        if bw_j is not None:
                            tv_j.bandwidth = bw_j
                        tv_j.fit_all(x_normed[:, [col]], y_residual, t_list=self.t_)
                        if self.options.smooth_coefs:
                            tv_j.smooth_coefs()

                        new_coef = tv_j.all_W.flatten()
                        change = np.max(np.abs(new_coef - tv_coefs_k[:, j]))
                        max_coef_change = max(max_coef_change, change)
                        tv_coefs_k[:, j] = new_coef

                    if max_coef_change < backfit_tol:
                        break

                tv_models_k = []
                for j, col in enumerate(tv_cols):
                    # Get bandwidth for this specific coefficient
                    bw_j = self._get_bandwidth_for_coefficient(bw_info, k, j) if bw_info is not None else None

                    tv_j = copy.deepcopy(self.tv_optimizer)
                    if self.init_conds is not None:
                        tv_j.initial_conditions = np.atleast_1d(self.init_conds[k,tv_idx][j])
                    if bw_j is not None:
                        tv_j.bandwidth = bw_j
                            
                    other_contrib = np.sum(x_normed[:, tv_cols] * tv_coefs_k, axis=1) - x_normed[:, col] * tv_coefs_k[:, j]
                    y_target = y_tv - other_contrib
                    tv_j.fit_all(x_normed[:, [col]], y_target, t_list=self.t_)

                    tv_models_k.append(tv_j)

                self.tv_models_[k] = tv_models_k
                self.tv_coefs_[k] = tv_coefs_k
                self.tv_biases_[k] = np.zeros(T) 
                self.coef_[k, tv_idx] = 0.0 
                
            if iteration == 0:
                pass
            else:
                max_tv_change = 0.0
                for k in range(n_targets):
                    if (self.tv_coefs_[k] is not None and 
                        self._prev_tv_coefs[k] is not None and
                        self.tv_coefs_[k].shape == self._prev_tv_coefs[k].shape):
                        max_tv_change = max(max_tv_change, 
                                        np.max(np.abs(self.tv_coefs_[k] - self._prev_tv_coefs[k])))
                max_change = max(max_change, max_tv_change)
                self._prev_tv_coefs = [c.copy() if c is not None else None for c in self.tv_coefs_]

            if max_change < 1e-8:
                break

        self._reconstruct_history([], initial_coef)
        if hasattr(self.base_optimizer, 'unbias') and self.base_optimizer.unbias:
            self._unbias_with_fixed(x_normed, y)
        if self.base_optimizer.normalize_columns:
            if self.fixed_values_ is not None and self.no_normalization_for_fixeds:
                fixed_bin = self.fixed_mask_.astype(int)
                non_fixed_bin = (~self.fixed_mask_).astype(int)
                self.coef_ = self.coef_ * fixed_bin + self.coef_ * non_fixed_bin / self.feat_norms_
            else:
                self.coef_ = self.coef_ / self.feat_norms_
            for i in range(len(self.history_)):
                self.history_[i] = self.history_[i] / self.feat_norms_

        self.intercept_ = 0.0
        self._is_fitted = True
        return self

    def _create_reduced(self, x, y_k, fixed_coef_k, target_idx):
        """(Устаревший метод, сохранён для совместимости, не используется)"""
        fixed_indices = self.fixed_mask_[target_idx]
        free_indices = ~fixed_indices
        y_fixed_contribution = np.zeros_like(y_k)
        if np.any(fixed_indices):
            y_fixed_contribution = x[:, fixed_indices] @ fixed_coef_k[fixed_indices]
        y_reduced = y_k - y_fixed_contribution
        self.reduced_indices_[target_idx] = np.where(free_indices)[0]
        x_reduced = x[:, free_indices]
        return x_reduced, y_reduced

    def _reconstruct_history(self, base_optimizers, initial_coef):
        """Reconstruct history from base optimizers (only constant parts)."""
        n_targets = len(base_optimizers)
        max_history_len = 0
        for opt in base_optimizers:
            if hasattr(opt, 'history_'):
                max_history_len = max(max_history_len, len(opt.history_))
        self.history_ = []
        for i in range(max_history_len):
            full_coef = initial_coef.copy()
            for k in range(n_targets):
                opt_k = base_optimizers[k]
                if (hasattr(opt_k, 'history_') and
                    i < len(opt_k.history_) and
                    self.reduced_indices_[k] is not None and
                    len(self.reduced_indices_[k]) > 0):
                    hist_coef_k = opt_k.history_[i]
                    full_coef[k, self.reduced_indices_[k]] = hist_coef_k
            self.history_.append(full_coef)
        self.ind_ = np.abs(self.coef_) > 1e-14

    def _unbias_with_fixed(self, x_normed, y):
        coef = self.coef_.copy()
        n_targets = y.shape[1]
        for i in range(n_targets):
            fixed_mask_i = self.fixed_mask_[i]
            tv_mask_i = self.tv_mask_[i]
            const_mask_i = (~fixed_mask_i) & (~tv_mask_i)
            active_const = const_mask_i & (np.abs(self.coef_[i]) > 1e-14)
            if not np.any(active_const):
                continue

            fixed_contrib = x_normed[:, fixed_mask_i] @ self.coef_[i, fixed_mask_i] if np.any(fixed_mask_i) else 0
            tv_contrib = 0
            if np.any(tv_mask_i) and self.tv_models_[i] is not None:
                tv_idx = np.where(tv_mask_i)[0]
                if self.tv_coefs_[i].shape[0] == x_normed.shape[0]:
                    tv_contrib = np.sum(x_normed[:, tv_idx] * self.tv_coefs_[i], axis=1) + self.tv_biases_[i]
                else:
                    print("here")
                    t_old = self.t_models[i] 
                    n_samples = x_normed.shape[0]
                    t_new = np.linspace(t_old[0], t_old[-1], n_samples)
                    interp_w = interp1d(t_old, self.tv_coefs_[i], axis=0, kind='linear', fill_value='extrapolate')
                    interp_b = interp1d(t_old, self.tv_biases_[i], kind='linear', fill_value='extrapolate')
                    tv_coef_interp = interp_w(t_new)
                    tv_bias_interp = interp_b(t_new)
                    tv_contrib = np.sum(x_normed[:, tv_idx] * tv_coef_interp, axis=1) + tv_bias_interp

            y_resid = y[:, i] - fixed_contrib - tv_contrib
            lr = LinearRegression(fit_intercept=False)
            lr.fit(x_normed[:, active_const], y_resid)
            coef[i, active_const] = lr.coef_
        self.coef_ = coef

    def predict(self, x_, t=None):
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        x = np.asarray(x_)
        n_samples = x.shape[0]
        if t is None:
            t = np.arange(n_samples)               
        t = np.asarray(t)

        y_pred = x @ self.coef_.T
        for k in range(len(self.tv_models_)):
            if self.tv_models_[k] is None:
                continue
            tv_idx = self.tv_mask_[k]
            if not np.any(tv_idx):
                continue
            tv_cols = np.where(tv_idx)[0]
            for j, col in enumerate(tv_cols):
                model_j = self.tv_models_[k][j]
                for i, ti in enumerate(t):
                    y_pred[i, k] += model_j.predict(ti, x[i, [col]].reshape(1, -1)).item()
        return y_pred
    
    def score(self, x_, y):
        """Return the coefficient of determination R^2."""
        y_pred = self.predict(x_)
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr, axis=0)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - ss_res / ss_tot

    def __getattr__(self, name):
        """Delegate unknown attributes to base optimizer."""
        if name in self.__dict__:
            return self.__dict__[name]
        elif hasattr(self.base_optimizer, name):
            return getattr(self.base_optimizer, name)
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")