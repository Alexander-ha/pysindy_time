import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pysindy.optimizers.fixed_base import FixedCoefficientOptimizer
from pysindy.SINDY_timevar.regressors.time_regressor import LassoTimeRegression
from pysindy.optimizers import STLSQ

# ============================================================================
dt = 0.005
t = np.arange(0, 10.0, dt)

c_true = lambda t: 1 / (1 + np.exp(-(t - 5)))
c1_true = lambda t: np.sin(0.8 * t)

def system(t, z):
    x, y = z
    return [c1_true(t) * y + 0.55*x, c_true(t) * x - 0.84*y]

sol = solve_ivp(system, [0, 10.0], [1.0, 0.5], t_eval=t)
x_clean = sol.y.T

x_dot = np.zeros_like(x_clean)
for j in range(2):
    spl = UnivariateSpline(t, x_clean[:, j], s=0)
    x_dot[:, j] = spl.derivative()(t)

x, y = x_clean[:, 0], x_clean[:, 1]

Theta = np.column_stack([
    x, y, x**2, y**2, x*y, x**3, y**3, x**2*y, x*y**2,
    np.sin(x), np.cos(x), np.sin(y), np.cos(y)
])

names = ['x', 'y', 'x²', 'y²', 'xy', 'x³', 'y³', 'x²y', 'xy²',
         'sin(x)', 'cos(x)', 'sin(y)', 'cos(y)']
n_features = Theta.shape[1]
print(f"Feature library size: {n_features}")

# ============================================================================
fixed_coefs = np.zeros((2, n_features), dtype=bool)
fixed_values = np.zeros((2, n_features))
time_varying_coefs = np.zeros((2, n_features), dtype=bool)

time_varying_coefs[0, 1] = True
time_varying_coefs[1, 0] = True 

for i in range(n_features):
    if i not in [0, 1]:
        fixed_coefs[0, i] = True
        fixed_coefs[1, i] = True
        fixed_values[0, i] = 0
        fixed_values[1, i] = 0

init_conds = np.zeros((2, n_features))
init_conds[0, 1] = np.sin(0.0)
init_conds[1, 0] = 1 / (1 + np.exp(5))

def epanechnikov_kernel(u):
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

#training via separated models
# dx/dt model (c1(t) - sine)
tv_optimizer_dx = LassoTimeRegression(
    iterations=2000, l1_penalty=0.01, bandwidth=0.8,
    kernel=epanechnikov_kernel, fit_intercept=False,
    use_prior=True, tau=50.0, prior_indices=[0]
)

model_dx = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=0.01, normalize_columns=False),
    fixed_coefs=fixed_coefs[0:1, :], fixed_values=fixed_values[0:1, :],
    time_varying_coefs=time_varying_coefs[0:1, :],
    tv_optimizer=tv_optimizer_dx, no_normalization_for_fixeds=True,
    init_conds=init_conds[0:1, :],
    options={'use_selector': False, 'smooth_coefs': True}
)
model_dx.max_iter = 80
model_dx.fit(Theta, x_dot[:, 0].reshape(-1, 1), t=t)

# dy/dt model (c(t) - sigmoid)
tv_optimizer_dy = LassoTimeRegression(
    iterations=2000, l1_penalty=1e-4, bandwidth=2.5,
    kernel=epanechnikov_kernel, fit_intercept=False,
    use_prior=True, tau=1000.0, prior_indices=[0]
)

model_dy = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=0.01, normalize_columns=False),
    fixed_coefs=fixed_coefs[1:2, :], fixed_values=fixed_values[1:2, :],
    time_varying_coefs=time_varying_coefs[1:2, :],
    tv_optimizer=tv_optimizer_dy, no_normalization_for_fixeds=True,
    init_conds=init_conds[1:2, :],
    options={'use_selector': False, 'smooth_coefs': True}
)
model_dy.max_iter = 120
model_dy.fit(Theta, x_dot[:, 1].reshape(-1, 1), t=t)

# ============================================================================
c1_est = model_dx.tv_coefs_[0][:, 0]
c_est = model_dy.tv_coefs_[0][:, 0]
const_dx = model_dx.coef_[0, 0]
const_dy = model_dy.coef_[0, 1]

# Trajectory reconstruction
def reconstruct(t_val, z, t_arr, c_vals, c1_vals):
    idx = np.argmin(np.abs(t_arr - t_val))
    x_val, y_val = z
    return [c1_vals[idx] * y_val + const_dx * x_val,
            c_vals[idx] * x_val + const_dy * y_val]

sol_rec=solve_ivp(lambda tv, zv: reconstruct(tv, zv, t, c_est, c1_est),[t[0], t[-1]], [1.0, 0.5], t_eval=t)
x_rec=sol_rec.y.T

#compute metrics on pred
r2_c = r2_score(c_true(t), c_est)
r2_c1 = r2_score(c1_true(t), c1_est)
mse_traj = mean_squared_error(x_clean, x_rec)
mse_c = mean_squared_error(c_true(t), c_est)
mse_c1 = mean_squared_error(c1_true(t), c1_est)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"c(t) (sigmoid):   R² = {r2_c:.4f},   MSE = {mse_c:.2e}")
print(f"c₁(t) (sine):     R² = {r2_c1:.4f},  MSE = {mse_c1:.2e}")
print(f"Trajectory:        MSE = {mse_traj:.2e}")
print(f"\nConstant coefficients:")
print(f"  dx/dt (x): {const_dx:.6f} (true: 0.55)")
print(f"  dy/dt (y): {const_dy:.6f} (true: -0.84)")
print(f"Sparsity: {100 * (1 - np.sum(np.abs(model_dx.coef_) > 1e-6) / n_features):.1f}%")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

axes[0, 0].plot(t, c1_true(t), 'k-', lw=2, label='True')
axes[0, 0].plot(t, c1_est, 'r--', lw=1.5, label='Estimated')
axes[0, 0].set_title(f'$c_1(t) = \\sin(0.8t)$\n$R^2 = {r2_c1:.4f}$')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('$c_1(t)$')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, c_true(t), 'k-', lw=2, label='True')
axes[0, 1].plot(t, c_est, 'r--', lw=1.5, label='Estimated')
axes[0, 1].set_title(f'$c(t) = 1/(1+e^{{-(t-5)}})$\n$R^2 = {r2_c:.4f}$')
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('$c(t)$')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(t, c1_true(t) - c1_est, 'b-', lw=1, alpha=0.7, label='$c_1(t)$ error')
axes[0, 2].plot(t, c_true(t) - c_est, 'r-', lw=1, alpha=0.7, label='$c(t)$ error')
axes[0, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 2].set_title('Reconstruction errors')
axes[0, 2].set_xlabel('t')
axes[0, 2].set_ylabel('Error')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].plot(x_clean[:, 0], x_clean[:, 1], 'k-', lw=2, label='True')
axes[1, 0].plot(x_rec[:, 0], x_rec[:, 1], 'r--', lw=1.5, label='Reconstructed')
axes[1, 0].set_title(f'Phase portrait\nMSE = {mse_traj:.2e}')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

axes[1, 1].plot(t, x_clean[:, 0], 'k-', lw=1.5, alpha=0.8, label='$x(t)$ true')
axes[1, 1].plot(t, x_rec[:, 0], 'r--', lw=1.2, label='$x(t)$ rec.')
axes[1, 1].plot(t, x_clean[:, 1], 'b-', lw=1.5, alpha=0.8, label='$y(t)$ true')
axes[1, 1].plot(t, x_rec[:, 1], 'g--', lw=1.2, label='$y(t)$ rec.')
axes[1, 1].set_title('Time series')
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('x, y')
axes[1, 1].legend(loc='upper right')
axes[1, 1].grid(True, alpha=0.3)

constants_true = [0.55, -0.84]
constants_est = [const_dx, const_dy]
x_pos = [0, 1]
axes[1, 2].bar(x_pos, constants_true, width=0.35, label='True', alpha=0.7, color='gray')
axes[1, 2].bar([p + 0.35 for p in x_pos], constants_est, width=0.35, label='Estimated', alpha=0.7, color='steelblue')
axes[1, 2].set_xticks([p + 0.175 for p in x_pos])
axes[1, 2].set_xticklabels(['$\\alpha$ (dx/dt, x)', '$\\beta$ (dy/dt, y)'])
axes[1, 2].set_ylabel('Coefficient value')
axes[1, 2].set_title('Constant coefficients')
axes[1, 2].legend()
axes[1, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('reconstruction_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: reconstruction_results.png")
plt.show()

