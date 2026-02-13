import copy
import numpy as np
from typing import Optional, Tuple, Union, List
from pysindy.optimizers.base import BaseOptimizer
from pysindy.utils import AxesArray, drop_nan_samples
from sklearn.utils.validation import check_X_y
from sklearn.linear_model import LinearRegression
from pysindy.optimizers.base import _preprocess_data, _normalize_features


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
    """
    
    def __init__(self, base_optimizer: BaseOptimizer,fixed_coefs: Optional[np.ndarray] = None, fixed_values: Optional[np.ndarray] = None, no_normalization_for_fixeds: Optional[bool] = True):
        self.base_optimizer = copy.deepcopy(base_optimizer) #copy base optimizer with all it's abilites
        self.fixed_coefs = fixed_coefs #apply fixed coefs
        self.fixed_values = fixed_values #apply fixed values
        self._is_fitted = False
        self.no_normalization_for_fixeds = no_normalization_for_fixeds


    
    def _init_fix_mask(self, coef_shape: Tuple[int, int]):
        """initialization of mask"""
        if self.fixed_coefs is None: 
            self.fixed_mask_ = np.zeros(coef_shape, dtype=bool) #initialize with the shape of coefs
            self.fixed_values_ = None #initially no fixed values
            return
        fixed_coefs = np.asarray(self.fixed_coefs) #make them as np
        # case 1 !!! - 1d input
        if fixed_coefs.ndim == 1: 
            if coef_shape[0] == 1: #apply the same dim
                fixed_coefs = fixed_coefs.reshape(1, -1) #provide 1 row and columns are arbitrary amount of length for our array
            else:
                raise ValueError(
                    f"fixed_coefs is 1d with shape {fixed_coefs.shape}, "
                    f"expected 2d!! {coef_shape} for {coef_shape[0]} amount of outputs")
        
        self.fixed_mask_ = fixed_coefs.astype(bool)
        if self.fixed_mask_.shape != coef_shape:
            raise ValueError(
                f"Shape of fixed_coefs {self.fixed_mask_.shape} "
                f"does not match expected shape {coef_shape}" )
        
        # init fixed vals 
        if self.fixed_values is not None:
            fixed_vals = np.asarray(self.fixed_values)
            if fixed_vals.ndim == 1 and coef_shape[0] == 1:
                fixed_vals = fixed_vals.reshape(1, -1)
            self.fixed_values_ = np.where(self.fixed_mask_, fixed_vals, 0) #now we fix all those vals that are about to be fixed here
        else:
            self.fixed_values_ = None
    


    def fit(self, x_, y, sample_weight=None, **reduce_kws):
        """
        fir with fixed coefficients
        
        Parameters
        ----------
        x_ : array-like, shape (n_samples, n_features) or (n_samples, n_features, n_trajectories)
            lib matrix
        y : array-like, shape (n_samples,) or (n_samples, n_targets)
            targeted vals
        sample_weight : array-like, optional
            sample weights
        reduce_kws : dict
            Additional arguments for base optimizer's _reduce method
            
        Returns
        -------
        self : object
        """
        # convert features and vals to arrays
        x_arr = np.asarray(x_)
        y_arr = np.asarray(y)
        
        # check accordint to dimmensionality 
        if x_arr.ndim == 2:
            x_axes = {"ax_sample": 0, "ax_coord": 1} 
        elif x_arr.ndim == 3:
            x_axes = {"ax_sample": 0, "ax_coord": 1, "ax_trajectory": 2}
        else:
            raise ValueError(
                f"x_ must be 2D or 3D array, got {x_arr.ndim}D with shape {x_arr.shape}" #not compatible
            )
        x_ = AxesArray(x_arr, x_axes) #adjust axes, reshape by tuple info 
        # the same technic more ways of organization
        if y_arr.ndim == 1:
            y_axes = {"ax_sample": 0}
        elif y_arr.ndim == 2:
            y_axes = {"ax_sample": 0, "ax_coord": 1}
        elif y_arr.ndim == 3:
            y_axes = {"ax_sample": 0, "ax_coord": 1, "ax_trajectory": 2}
        else:
            raise ValueError(
                f"y must be 1D, 2D or 3D array, got {y_arr.ndim}D with shape {y_arr.shape}"
            )
        y = AxesArray(y_arr, y_axes) #adjust axes, reshape by tuple info
        x_, y = drop_nan_samples(x_, y) #clean nan sample by default method from sindy
        x_, y = check_X_y(x_, y, accept_sparse=[], y_numeric=True, multi_output=True) #check also, 
        
        # Preprocessing
        x, y, X_offset, y_offset, _, sample_weight_sqrt = _preprocess_data(x_,y,fit_intercept=False,copy=self.base_optimizer.copy_X,sample_weight=sample_weight,)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1) #reshape adjusting size of rows 
        
        coef_shape = (y.shape[1], x.shape[1]) #now we have shape
        self._init_fix_mask(coef_shape) #use fixing mask and prepare of reduced problem
        x_normed = np.copy(x)
        if self.base_optimizer.normalize_columns:
            feat_norms, x_normed = _normalize_features(x_normed)
            self.feat_norms_ = feat_norms
        
        # init coefficients
        if self.base_optimizer.initial_guess is None:
            initial_coef = np.linalg.lstsq(x_normed, y, rcond=None)[0].T #apriori 
        else:
            initial_coef = self.base_optimizer.initial_guess
        self.coef_ = initial_coef.copy()
        if self.fixed_values_ is not None:
            self.coef_[self.fixed_mask_] = self.fixed_values_[self.fixed_mask_]
        
        # pick indices from reduced problem
        n_targets = y.shape[1]
        self.reduced_indices_ = [None] * n_targets
        base_optimizers = []
        for k in range(n_targets):
            x_reduced_k, y_reduced_k = self._create_reduced(x_normed, y[:, k], self.coef_[k], k) #reduced problem created
            opt_k = copy.deepcopy(self.base_optimizer) #base copy
            opt_k.fit(x_reduced_k, y_reduced_k,sample_weight=sample_weight,  **reduce_kws ) #fix on reduced ONLY(!)
            base_optimizers.append(opt_k)
            if self.reduced_indices_[k] is not None and len(self.reduced_indices_[k]) > 0:
                if isinstance(opt_k.coef_, np.ndarray) and opt_k.coef_.ndim == 2:
                    if opt_k.coef_.shape[0] > 0:
                        self.coef_[k, self.reduced_indices_[k]] = opt_k.coef_[0]#get coefs and apply them to self coefs of current model
                else:
                    self.coef_[k, self.reduced_indices_[k]] = opt_k.coef_
        
        # RECOVER: History from each optimizer of problem
        self._reconstruct_history(base_optimizers, initial_coef)
        
        if hasattr(self.base_optimizer, 'unbias') and self.base_optimizer.unbias and np.any(self.ind_):
            self._unbias_with_fixed(x_normed, y)
        
        # revert normalization if applied
        if self.base_optimizer.normalize_columns:
            if self.fixed_values_ is not None and self.no_normalization_for_fixeds is True:
                fixed_bin = self.fixed_mask_.astype(int)
                non_fixed_bin = (~self.fixed_mask_).astype(int)
                self.coef_ = self.coef_*fixed_bin + self.coef_*non_fixed_bin / self.feat_norms_
            else:
                self.coef_ = self.coef_ / self.feat_norms_
            for i in range(len(self.history_)):
                self.history_[i] = self.history_[i] / self.feat_norms_
        
        self.intercept_ = 0.0
        self._is_fitted = True
        #if (self.base_optimizer.normalize == True and self.perform_denormalization == True):
        #   self.coef
        return self
    
    def _create_reduced(self, x: np.ndarray, y_k: np.ndarray,fixed_coef_k: np.ndarray, target_idx: int):
        """Create reduced optimization problem for a single target."""
        fixed_indices = self.fixed_mask_[target_idx]
        free_indices = ~fixed_indices
        y_fixed_contribution = np.zeros_like(y_k)
        if np.any(fixed_indices):
            y_fixed_contribution = x[:, fixed_indices] @ fixed_coef_k[fixed_indices]
        
        # now we want to minim only those that are not contributing in fixed vals
        y_reduced = y_k - y_fixed_contribution
        # Store reduced indices for target
        self.reduced_indices_[target_idx] = np.where(free_indices)[0]
        # build matrix
        x_reduced = x[:, free_indices]
        
        return x_reduced, y_reduced
    
    def _reconstruct_history(self, base_optimizers: List[BaseOptimizer], 
                            initial_coef: np.ndarray):
        """Reconstruct history from multiple base optimizers."""
        n_targets = len(base_optimizers)
        
        # Find maximum history length
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
                    
                    # Update full coefficient matrix (only non-fixed coefficients)
                    full_coef[k, self.reduced_indices_[k]] = hist_coef_k
            
            self.history_.append(full_coef)
        
        # Set indicator for non-zero coefficients
        self.ind_ = np.abs(self.coef_) > 1e-14
    
    def _unbias_with_fixed(self, x: np.ndarray, y: np.ndarray):
        """Apply unbiasing only to non-fixed coefficients."""
        coef = self.coef_.copy()
        
        for i in range(self.ind_.shape[0]):
            free_and_active = ~self.fixed_mask_[i] & self.ind_[i]
            if np.any(free_and_active):
                try:
                    lr = LinearRegression(fit_intercept=False)
                    lr.fit(x[:, free_and_active], y[:, i])
                    coef[i, free_and_active] = lr.coef_
                except:
                    # save coefs if regressor dies
                    pass
        
        self.coef_ = coef
    
    def predict(self, x_):
        """Predict using the linear model."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")
        x = np.asarray(x_)
        if hasattr(self, 'feat_norms_') and self.base_optimizer.normalize_columns:
            x = x / self.feat_norms_
        
        return x @ self.coef_.T
    
    def score(self, x_, y):
        """Return the coefficient of determination R^2."""
        y_pred = self.predict(x_)
        y_arr = np.asarray(y)
        
        # 1d case score
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