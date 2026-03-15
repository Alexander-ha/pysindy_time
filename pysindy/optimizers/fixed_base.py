import copy
import numpy as np
from typing import Optional, Tuple, List
from pysindy.optimizers.base import BaseOptimizer
from pysindy.utils import AxesArray, drop_nan_samples
from sklearn.utils.validation import check_X_y
from sklearn.linear_model import LinearRegression
from pysindy.optimizers.base import _preprocess_data, _normalize_features
from pysindy.SINDY_timevar.regressors.time_regressor import LassoTimeRegression


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
        if True, fixed coefficients are not denormalized (only relevant if normalize_columns=True).
    """
    def __init__(self, base_optimizer: BaseOptimizer,
                 fixed_coefs: Optional[np.ndarray] = None,
                 fixed_values: Optional[np.ndarray] = None,
                 time_varying_coefs: Optional[np.ndarray] = None,
                 tv_optimizer_params: Optional[dict] = None,
                 tv_optimizer: Optional[LassoTimeRegression] = None,
                 no_normalization_for_fixeds: Optional[bool] = True):
        self.base_optimizer = copy.deepcopy(base_optimizer)
        self.fixed_coefs = fixed_coefs
        self.fixed_values = fixed_values
        self.time_varying_coefs = time_varying_coefs
        self.no_normalization_for_fixeds = no_normalization_for_fixeds
        self._is_fitted = False
        self.max_iter = 50

        if tv_optimizer is not None:
            self.tv_optimizer = tv_optimizer
        else:
            tv_optimizer_params = tv_optimizer_params or {}
            self.tv_optimizer = LassoTimeRegression(**tv_optimizer_params)

    def _init_fix_mask(self, coef_shape: Tuple[int, int]):
        """Initialization of both fixed and time‑varying masks."""
        # Fixed mask
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

        # Time‑varying mask
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


        for iteration in range(self.max_iter):
            for k in range(n_targets):
                fixed_idx = self.fixed_mask_[k]
                tv_idx = self.tv_mask_[k]
                const_idx = ~(fixed_idx | tv_idx)
                y_k = y[:, k].copy()
                if np.any(fixed_idx):
                    y_k -= x_normed[:, fixed_idx] @ self.coef_[k, fixed_idx]

                if np.any(tv_idx) and self.tv_models_[k] is not None:
                    tv_contrib = np.array([
                        self.tv_models_[k].predict(ti, x_normed[i, tv_idx].reshape(1, -1))
                        for i, ti in enumerate(self.t_)
                    ]).flatten()
                    y_k -= tv_contrib

                if np.any(const_idx):
                    x_red = x_normed[:, const_idx]
                    opt_k = copy.deepcopy(self.base_optimizer)
                    opt_k.fit(x_red, y_k, sample_weight=sample_weight, **reduce_kws)
                    self.coef_[k, const_idx] = opt_k.coef_.flatten()

                if np.any(tv_idx):
                    y_tv = y[:, k].copy()
                    if np.any(fixed_idx):
                        y_tv -= x_normed[:, fixed_idx] @ self.coef_[k, fixed_idx]
                    if np.any(const_idx):
                        y_tv -= x_normed[:, const_idx] @ self.coef_[k, const_idx]

                    tv_model = copy.deepcopy(self.tv_optimizer)
                    tv_model.fit_all(x_normed[:, tv_idx], y_tv, t_list=self.t_)
                    self.tv_models_[k] = tv_model
                    self.tv_coefs_[k] = tv_model.all_W
                    self.tv_biases_[k] = tv_model.all_b
                    self.coef_[k, tv_idx] = 0.0
        

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
                    pass

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

            tv_model = self.tv_models_[k]
            for i, ti in enumerate(t):
                y_pred[i, k] += tv_model.predict(ti, x[i, tv_idx].reshape(1, -1)).item()
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