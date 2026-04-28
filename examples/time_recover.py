import sys
import os
import numpy as np


os.makedirs('saves', exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pysindy.optimizers.fixed_base import FixedCoefficientOptimizer
from pysindy.SINDY_timevar.regressors.time_regressor import LassoTimeRegression
from pysindy.feature_library import PolynomialLibrary
from pysindy.differentiation import FiniteDifference
from pysindy.optimizers import STLSQ
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import ParameterGrid
import itertools

dt = 0.001
t = np.arange(0, 1, dt)
N_train = int(len(t) * 0.5)
N_val = int(len(t) * 0.3)
N_test = len(t) - N_train - N_val

t_train = t[:N_train]
t_val = t[N_train:N_train+N_val]
t_test = t[N_train+N_val:]

def time_coeff(t):
    return 1 + 1 / (1 + np.exp(-10 * (t - 0.5)))

coeff_true = time_coeff(t)

def ode_system(t, x, coeff_func):
    c = coeff_func(t)
    return [-2.0 * c * x[0],
             c * x[1]]

x0 = [3.0, 0.5]

sol = solve_ivp(lambda t, x: ode_system(t, x, time_coeff),[t[0], t[-1]],x0, t_eval=t, method='RK45')
x_data = sol.y.T

x_train = x_data[:N_train, :]
x_val = x_data[N_train:N_train+N_val, :]
x_test = x_data[N_train+N_val:, :]

diff = FiniteDifference(order=2)
x_dot_train = diff._differentiate(x_train, t_train)
x_dot_val = diff._differentiate(x_val, t_val)

library = PolynomialLibrary(degree=2, include_bias=True)
library.fit(x_train)
Theta_train = library.transform(x_train)
Theta_val = library.transform(x_val)
feature_names = library.get_feature_names()

n_features = Theta_train.shape[1]
print("Feature names:", feature_names)

n_targets = 2
fixed_mask = np.zeros((n_targets, n_features), dtype=bool)
tv_mask = np.zeros((n_targets, n_features), dtype=bool)
fixed_values = np.zeros((n_targets, n_features))

tv_mask[0, 1] = True
for j in range(n_features):
    if j != 1:
        fixed_mask[0, j] = True
tv_mask[1, 2] = True
for j in range(n_features):
    if j != 2:
        fixed_mask[1, j] = True


epanechnikov = lambda u: 0.75 * (1 - u**2) * (np.abs(u) <= 1)

param_grid = {
    'bandwidth': [0.01, 0.05, 0.1, 0.2, 0.5],
    'l1_penalty': [0.0001, 0.001, 0.01, 0.1],
    'iterations': [1000],
}

best_score = -np.inf
best_params = None
best_model = None

print("Grid Search...")
total_combinations = len(list(ParameterGrid(param_grid)))
current = 0

for params in ParameterGrid(param_grid):
    current += 1
    print(f"{current}/{total_combinations}: {params}")
    
    tv_opt = LassoTimeRegression(
        iterations=params['iterations'],
        l1_penalty=params['l1_penalty'],
        bandwidth=params['bandwidth'],
        kernel=epanechnikov
    )
    
    base_opt = STLSQ(threshold=0.01, normalize_columns=False)
    
    proxy_opt = FixedCoefficientOptimizer(
        base_optimizer=base_opt,
        fixed_coefs=fixed_mask,
        fixed_values=fixed_values,
        time_varying_coefs=tv_mask,
        tv_optimizer=tv_opt,
        no_normalization_for_fixeds=True
    )
    
    proxy_opt.fit(Theta_train, x_dot_train)
    
    def compute_validation_score(opt, Theta_val, x_dot_val, t_val):
        total_mse = 0
        coef_np = np.asarray(opt.coef_)
        for k in range(2):
            tv_model = opt.tv_models_[k]
            if tv_model is None:
                continue
            tv_idx = np.where(opt.tv_mask_[k])[0]
            if len(tv_idx) == 0:
                continue
            y_pred = np.zeros_like(x_dot_val[:, k])
            t_values = tv_model.t_values_
            tv_coefs = opt.tv_coefs_[k]
            tv_biases = opt.tv_biases_[k] 
            for i, ti in enumerate(t_val):
                idx = np.argmin(np.abs(t_values - ti))
                tv_coef = tv_coefs[idx]
                tv_bias = tv_biases[idx]
                theta_i = np.asarray(Theta_val[i]).flatten()
                const_part = coef_np[k] @ theta_i
                tv_part = theta_i[tv_idx] @ tv_coef + tv_bias 
                y_pred[i] = const_part + tv_part
            mse = mean_squared_error(x_dot_val[:, k], y_pred)
            total_mse += mse
        return -total_mse    
    score = compute_validation_score(proxy_opt, Theta_val, x_dot_val, t_val)
    
    if score > best_score:
        best_score = score
        best_params = params
        best_model = proxy_opt
        print(f"  new best score! Score: {score:.6f}")

print(f"\nbest params: {best_params}")
print(f"N-MSE BEST SCORE: {best_score:.6f}")

print("\nFinal model learning with new h-params...")

x_train_full = x_data[:N_train+N_val, :]
t_train_full = t[:N_train+N_val]
x_dot_train_full = diff._differentiate(x_train_full, t_train_full)
Theta_train_full = library.transform(x_train_full)

tv_opt_best = LassoTimeRegression(**best_params, kernel=epanechnikov)
base_opt = STLSQ(threshold=1e-8, normalize_columns=False)

final_model = FixedCoefficientOptimizer(
    base_optimizer=base_opt,
    fixed_coefs=fixed_mask,
    fixed_values=fixed_values,
    time_varying_coefs=tv_mask,
    tv_optimizer=tv_opt_best,
    no_normalization_for_fixeds=True
)


final_model.fit(Theta_train_full, x_dot_train_full, t=t_train_full)
print("Shape of coef_:", final_model.coef_.shape)
print("Fixed mask (0):", final_model.fixed_mask_[0])
print("Fixed values (0):", final_model.fixed_values_[0])
print("Coef after fit (0):", final_model.coef_[0])
print("TV mask (0):", final_model.tv_mask_[0])

print("\n" + "="*50)
print("DIAGNOSTICS")
print("="*50)

for k in range(2):
    if final_model.tv_models_[k] is not None:
        tv_coefs = final_model.tv_coefs_[k]
        print(f"\nTarget {k} TV coefficients:")
        print(f"  Shape: {tv_coefs.shape}")
        print(f"  Mean: {np.mean(tv_coefs):.6f}")
        print(f"  Std: {np.std(tv_coefs):.6f}")
        print(f"  Min: {np.min(tv_coefs):.6f}")
        print(f"  Max: {np.max(tv_coefs):.6f}")
        print(f"  Non-zero: {np.sum(np.abs(tv_coefs) > 1e-6)} / {tv_coefs.size}")

print("\nConstant coefficients:")
print(final_model.coef_)
print(f"  L2 norm: {np.linalg.norm(final_model.coef_):.6f}")
print(f"  Non-zero: {np.sum(np.abs(final_model.coef_) > 1e-6)} / {final_model.coef_.size}")

sample_idx = 0
t_sample = t_train_full[sample_idx]
x_sample = Theta_train_full[sample_idx:sample_idx+1]
true_val = x_dot_train_full[sample_idx]

const_part = x_sample @ final_model.coef_.T
tv_part = np.zeros(2)
for k in range(2):
    if final_model.tv_models_[k] is not None:
        tv_idx = final_model.tv_mask_[k]
        tv_part[k] = final_model.tv_models_[k].predict(
            t_sample, x_sample[0, tv_idx].reshape(1, -1)
        ).item()

print(f"\nSample prediction decomposition:")
print(f"  True value: {true_val}")
print(f"  Constant part: {const_part[0]}")
print(f"  TV part: {tv_part}")
print(f"  Total: {const_part[0] + tv_part}")
plt.figure(figsize=(10, 4))

for k in range(2):
    tv_model = final_model.tv_models_[k]
    if tv_model is not None:
        t_values = tv_model.t_values_
        tv_coefs = final_model.tv_coefs_[k]
        if k == 0:
            print("Coef time-varying x0:")
            print(f"max: {tv_coefs[:,0].max()}, min: {tv_coefs[:,0].min()}, mean: {tv_coefs[:,0].mean()}")
            plt.plot(t_values, tv_coefs[:, 0], label='tv-var coef estimation dx/dt x0 ')
            plt.plot(t, -2 * coeff_true, '--', label='real: -2*c(t)')
        else:
            plt.plot(t_values, tv_coefs[:, 0], label='tv-var coef estim dy/dt x1')
            plt.plot(t, coeff_true, '--', label='real: c(t)')
plt.xlabel('t')
plt.ylabel('coef')
plt.legend()
plt.title('comparison of coefs real and timevar recover (best params)')
plt.savefig('saves/coefficients_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

def rhs_with_tv(t, x):
    x_row = x.reshape(1, -1)
    Theta = np.asarray(library.transform(x_row)).flatten()
    coef = np.asarray(final_model.coef_)
    
    deriv = coef @ Theta
    
    for k in range(2):
        tv_model = final_model.tv_models_[k]
        if tv_model is None:
            continue
        tv_idx = np.where(final_model.tv_mask_[k])[0]
        if len(tv_idx) == 0:
            continue
        
        t_values = tv_model.t_values_
        tv_coefs = final_model.tv_coefs_[k]
        tv_biases = final_model.tv_biases_[k]
        i = np.argmin(np.abs(t_values - t))
        tv_coef = tv_coefs[i]
        tv_bias = tv_biases[i]
        deriv[k] += Theta[tv_idx] @ tv_coef + tv_bias
    
    return deriv

sol_model = solve_ivp(rhs_with_tv, [t[0], t[-1]], x0, t_eval=t, method='RK45')
x_model = sol_model.y.T

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(t, x_data[:, 0], 'k', linewidth=1, label='real x(t)')
plt.plot(t, x_model[:, 0], 'r--', linewidth=1.5, label=' SINDy (x)')
plt.axvline(x=t[N_train+N_val], color='gray', linestyle=':', label='test start')
plt.xlabel('t')
plt.ylabel('x')
plt.legend()
plt.title('x')

plt.subplot(1, 2, 2)
plt.plot(t, x_data[:, 1], 'k', linewidth=1, label='real y(t)')
plt.plot(t, x_model[:, 1], 'b--', linewidth=1.5, label=' SINDy (y)')
plt.axvline(x=t[N_train+N_val], color='gray', linestyle=':', label='test start')
plt.xlabel('t')
plt.ylabel('y')
plt.legend()
plt.title('y')

plt.tight_layout()
plt.savefig('saves/trajectories_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

r2_x = r2_score(x_test[:, 0], x_model[N_train+N_val:, 0])
r2_y = r2_score(x_test[:, 1], x_model[N_train+N_val:, 1])
print(final_model.coef_)

for k, tv_model in enumerate(final_model.tv_models_):
    if tv_model is not None:
        print(f"Target {k}: t_values_ length = {len(tv_model.t_values_)}")

train_pred = final_model.predict(Theta_train_full, t=t_train_full)
train_resid = x_dot_train_full - train_pred
print("Training residual norm:", np.linalg.norm(train_resid))

import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.plot(t_train_full, x_dot_train_full[:,0], label='true dx/dt')
plt.plot(t_train_full, train_pred[:,0], '--', label='pred dx/dt')
plt.legend()
plt.title('Derivative comparison on training data')
plt.show()

print(f"R^2 x: {r2_x:.4f}")
print(f"R^2 y: {r2_y:.4f}")

print(f"\nbest params for LassoTimeRegression:")
print(f"  bandwidth: {best_params['bandwidth']}")
print(f"  l1_penalty: {best_params['l1_penalty']}")
print(f"  iterations: {best_params['iterations']}")
print(f"  learning_rate: {best_params['learning_rate']}")