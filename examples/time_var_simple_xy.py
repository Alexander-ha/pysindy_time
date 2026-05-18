import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score, mean_squared_error
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pysindy.optimizers.fixed_base import FixedCoefficientOptimizer
from pysindy.SINDY_timevar.regressors.time_regressor import LassoTimeRegression
from pysindy.optimizers import STLSQ

dt = 0.02
t = np.arange(0, 10.0, dt)

c1_true = lambda t: -(t - 5)**2 / 10 + 1.5
c2_true = lambda t: 0.8 * np.cos(1.2 * t) 
c3_true = lambda t: 0.5 * np.sin(1.5 * t)

def system(t, z):
    x, y = z
    return [c1_true(t) * y + c3_true(t) * x * y,
            c2_true(t) * x]

sol = solve_ivp(system, [0, 10.0], [1.0, 0.8], t_eval=t)
x_clean = sol.y.T
x_dot = np.zeros_like(x_clean)
for j in range(2):
    spl = UnivariateSpline(t, x_clean[:, j], s=1e-4)
    x_dot[:, j] = spl.derivative()(t)

x, y = x_clean[:, 0], x_clean[:, 1]
Theta = np.column_stack([
    y, x,                # 0: y for c1(t), 1: x (constant - should be zero)
    x, y,                # 2: x for c2(t), 3: y (constant - should be zero)
    x*y,                 # 4: xy with time-varying coefficient c3(t) (only in dx/dt)
    x**2, y**2,          # 5,6: quadratic (should be zero)
    x**3, y**3,          # 7,8: cubic (should be zero)
    np.sin(x), np.cos(x), np.sin(y), np.cos(y)  # 9-12: trig (should be zero)
])

names = ['y_tv', 'x_const', 'x_tv', 'y_const', 'xy_tv', 
         'x²', 'y²', 'x³', 'y³', 'sin(x)', 'cos(x)', 'sin(y)', 'cos(y)']
n_features = Theta.shape[1]

print("="*60)
print("DYNAMICAL SYSTEM")
print("="*60)
print("dx/dt = c₁(t)·y + c₃(t)·xy")
print("dy/dt = c₂(t)·x")
print(f"c₁(t) = -(t-5)²/10 + 1.5 (parabola)")
print(f"c₂(t) = 0.8·cos(1.2t)")
print(f"c₃(t) = 0.5·sin(1.5t) (xy coefficient - ONLY in dx/dt!)")
print("="*60)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
fixed_coefs = np.zeros((2, n_features), dtype=bool)
fixed_values = np.zeros((2, n_features))
time_varying_coefs = np.zeros((2, n_features), dtype=bool)

# Time-varying coefficients
time_varying_coefs[0, 0] = True   # dx/dt: c1(t) * y
time_varying_coefs[0, 4] = True   # dx/dt: c3(t) * xy
time_varying_coefs[1, 2] = True   # dy/dt: c2(t) * x

# NO coefficients are fixed to zero - let sparsity discovery work

print(f"\nTime-varying coefficients:")
print(f"  dx/dt: y (c₁), xy (c₃)")
print(f"  dy/dt: x (c₂)")
print(f"Optimizable constant coefficients: {2*n_features - np.sum(time_varying_coefs)}")
print("(These should become ZERO via L1 regularization)")

init_conds = np.zeros((2, n_features))
init_conds[0, 0] = c1_true(0)
init_conds[0, 4] = c3_true(0)
init_conds[1, 2] = c2_true(0)

def epanechnikov_kernel(u):
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

# ============================================================================
# TRAINING WITH FIXED BANDWIDTHS (NO GRID SEARCH)
# ============================================================================
print("\n" + "="*60)
print("TRAINING WITH FIXED BANDWIDTHS")
print("="*60)

# Manual bandwidths (no grid search)
bw_dx = 2.0
bw_dy = 1.5

tv_optimizer_dx = LassoTimeRegression(
    iterations=3000, l1_penalty=0.01, bandwidth=bw_dx,
    kernel=epanechnikov_kernel, fit_intercept=False,
    use_prior=True, tau=1000.0, prior_indices=[0, 1]
)

tv_optimizer_dy = LassoTimeRegression(
    iterations=3000, l1_penalty=0.01, bandwidth=bw_dy,
    kernel=epanechnikov_kernel, fit_intercept=False,
    use_prior=True, tau=1000.0, prior_indices=[0]
)

model_dx = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=0.001, normalize_columns=False),
    fixed_coefs=fixed_coefs[0:1, :], fixed_values=fixed_values[0:1, :],
    time_varying_coefs=time_varying_coefs[0:1, :],
    tv_optimizer=tv_optimizer_dx, no_normalization_for_fixeds=True,
    init_conds=init_conds[0:1, :],
    options={'use_selector': False, 'smooth_coefs': True}
)

model_dy = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=0.01, normalize_columns=False),
    fixed_coefs=fixed_coefs[1:2, :], fixed_values=fixed_values[1:2, :],
    time_varying_coefs=time_varying_coefs[1:2, :],
    tv_optimizer=tv_optimizer_dy, no_normalization_for_fixeds=True,
    init_conds=init_conds[1:2, :],
    options={'use_selector': False, 'smooth_coefs': True}
)

model_dx.max_iter = 20
model_dy.max_iter = 20
model_dx.fit(Theta, x_dot[:, 0].reshape(-1, 1), t=t)
model_dy.fit(Theta, x_dot[:, 1].reshape(-1, 1), t=t)

# Extract results
c1_est = model_dx.tv_coefs_[0][:, 0]
c3_est = model_dx.tv_coefs_[0][:, 1]
c2_est = model_dy.tv_coefs_[0][:, 0]

# Constant coefficients (should be zero)
const_dx = model_dx.coef_[0, :]
const_dy = model_dy.coef_[0, :]

# Smoothing
c1_smooth = gaussian_filter1d(c1_est, sigma=2)
c2_smooth = gaussian_filter1d(c2_est, sigma=2)
c3_smooth = gaussian_filter1d(c3_est, sigma=2)

# Trajectory reconstruction
def reconstruct(t_val, z, t_arr, c1_vals, c2_vals, c3_vals):
    idx = np.argmin(np.abs(t_arr - t_val))
    x_val, y_val = z
    dxdt = c1_vals[idx] * y_val + c3_vals[idx] * x_val * y_val
    dydt = c2_vals[idx] * x_val
    return [dxdt, dydt]

sol_rec = solve_ivp(lambda tv, zv: reconstruct(tv, zv, t, c1_smooth, c2_smooth, c3_smooth),
                    [t[0], t[-1]], [1.0, 0.8], t_eval=t)
x_rec = sol_rec.y.T

# Metrics
r2_c1 = r2_score(c1_true(t), c1_smooth)
r2_c2 = r2_score(c2_true(t), c2_smooth)
r2_c3 = r2_score(c3_true(t), c3_smooth)
mse_traj = mean_squared_error(x_clean, x_rec)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"Bandwidths: dx/dt={bw_dx}, dy/dt={bw_dy}")
print(f"c₁(t) (parabola):        R² = {r2_c1:.4f}")
print(f"c₂(t) (cosine):          R² = {r2_c2:.4f}")
print(f"c₃(t) (xy coefficient):  R² = {r2_c3:.4f}")
print(f"Trajectory MSE:          {mse_traj:.2e}")
print(f"\nConstant coefficients (should be ZERO):")
print(f"  dx/dt constants: {np.array2string(const_dx, precision=4, suppress_small=True)}")
print(f"  dy/dt constants: {np.array2string(const_dy, precision=4, suppress_small=True)}")
nonzero_const = np.sum(np.abs(const_dx) > 1e-6) + np.sum(np.abs(const_dy) > 1e-6)
print(f"Non-zero constants: {nonzero_const}")
print(f"Sparsity: {100*(1 - nonzero_const / (2*n_features)):.1f}%")
print(f"{'='*60}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# c₁(t)
axes[0, 0].plot(t, c1_true(t), 'k-', lw=2, label='True')
axes[0, 0].plot(t, c1_smooth, 'r--', lw=1.5, label='Estimated')
axes[0, 0].set_title(f'$c_1(t)$ (parabola)\n$R^2 = {r2_c1:.4f}$')
axes[0, 0].set_xlabel('t')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# c₂(t)
axes[0, 1].plot(t, c2_true(t), 'k-', lw=2, label='True')
axes[0, 1].plot(t, c2_smooth, 'r--', lw=1.5, label='Estimated')
axes[0, 1].set_title(f'$c_2(t)$ (cosine)\n$R^2 = {r2_c2:.4f}$')
axes[0, 1].set_xlabel('t')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# c₃(t) - xy coefficient (only in dx/dt)
axes[0, 2].plot(t, c3_true(t), 'k-', lw=2, label='True')
axes[0, 2].plot(t, c3_smooth, 'r--', lw=1.5, label='Estimated')
axes[0, 2].set_title(f'$c_3(t)$ (xy coefficient)\n$R^2 = {r2_c3:.4f}$')
axes[0, 2].set_xlabel('t')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Phase portrait
axes[1, 0].plot(x_clean[:, 0], x_clean[:, 1], 'k-', lw=2, label='True')
axes[1, 0].plot(x_rec[:, 0], x_rec[:, 1], 'r--', lw=1.5, label='Reconstructed')
axes[1, 0].set_title(f'Phase portrait\nMSE = {mse_traj:.2e}')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

# Time series
axes[1, 1].plot(t, x_clean[:, 0], 'k-', lw=1.5, alpha=0.8, label='$x(t)$ true')
axes[1, 1].plot(t, x_rec[:, 0], 'r--', lw=1.2, label='$x(t)$ rec.')
axes[1, 1].plot(t, x_clean[:, 1], 'b-', lw=1.5, alpha=0.8, label='$y(t)$ true')
axes[1, 1].plot(t, x_rec[:, 1], 'g--', lw=1.2, label='$y(t)$ rec.')
axes[1, 1].set_title('Time series')
axes[1, 1].set_xlabel('t')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Constant coefficients sparsity
all_const = np.concatenate([const_dx, const_dy])
colors = ['red' if abs(val) > 1e-6 else 'steelblue' for val in all_const]
axes[1, 2].bar(np.arange(len(all_const)), np.abs(all_const), alpha=0.7, color=colors)
axes[1, 2].set_xticks(np.arange(len(all_const)))
xticklabels = [f'{names[i]}' for i in range(n_features)] + [f'{names[i]}' for i in range(n_features)]
axes[1, 2].set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=8)
axes[1, 2].set_ylabel('|Coefficient|')
axes[1, 2].set_title('Constant coefficients (should be ZERO)')
axes[1, 2].axhline(y=1e-6, color='r', linestyle='--', label='Zero threshold (1e-6)')
axes[1, 2].set_yscale('log')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('tv_xy_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: tv_xy_results.png")
plt.show()

print("\n" + "="*60)
print("VERDICT")
print("="*60)
if r2_c1 > 0.9: print(f"✅ c₁(t): R² = {r2_c1:.4f}")
else: print(f"⚠️ c₁(t): R² = {r2_c1:.4f}")
if r2_c2 > 0.9: print(f"✅ c₂(t): R² = {r2_c2:.4f}")
else: print(f"⚠️ c₂(t): R² = {r2_c2:.4f}")
if r2_c3 > 0.9: print(f"✅ c₃(t): R² = {r2_c3:.4f}")
else: print(f"⚠️ c₃(t): R² = {r2_c3:.4f}")
print(f"✅ Constant coefficients: {nonzero_const} non-zero out of {2*n_features}")
print("="*60)