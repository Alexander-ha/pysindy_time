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

case = 1
noise_level = 0.05  # 5% noise

dt = 0.01
t = np.arange(0, 5.0, dt)

c_true = lambda t: 1 / (1 + np.exp(-(t - 5)))

if case == 1:
    print("Case 1: dx/dt = c(t)*y, dy/dt = c(t)*x (positive c(t))")
    def system(t, z):
        x, y = z
        c = c_true(t)
        return [c * y, c * x]
    const_dx_true = 0.0
    const_dy_true = 0.0
else:
    print("Case 2: dx/dt = -c(t)*y, dy/dt = -c(t)*x (negative -c(t))")
    def system(t, z):
        x, y = z
        c = -c_true(t)
        return [c * y, c * x]
    const_dx_true = 0.0
    const_dy_true = 0.0

# Generate clean data
sol = solve_ivp(system, [0, 10.0], [1.0, 0.5], t_eval=t)
x_clean = sol.y.T

# Add noise to states
np.random.seed(42)
x_noisy = x_clean + noise_level * np.random.randn(*x_clean.shape) * np.std(x_clean, axis=0)

# Compute derivatives from noisy data with smoothing
x_dot = np.zeros_like(x_noisy)
for j in range(2):
    spl = UnivariateSpline(t, x_noisy[:, j], s=len(t) * noise_level)
    x_dot[:, j] = spl.derivative()(t)

x, y = x_noisy[:, 0], x_noisy[:, 1]

Theta = np.column_stack([x, y])
names = ['x', 'y']
n_features = Theta.shape[1]
print(f"Feature library size: {n_features}")
print(f"Noise level: {noise_level * 100}%")

fixed_coefs = np.zeros((2, n_features), dtype=bool)
fixed_values = np.zeros((2, n_features))
time_varying_coefs = np.zeros((2, n_features), dtype=bool)

time_varying_coefs[0, 1] = True   # dx/dt: c(t) * y
time_varying_coefs[1, 0] = True   # dy/dt: c(t) * x
for i in range(n_features):
    if not time_varying_coefs[0, i]:
        fixed_coefs[0, i] = True
        fixed_values[0, i] = 0
    if not time_varying_coefs[1, i]:
        fixed_coefs[1, i] = True
        fixed_values[1, i] = 0

init_conds = np.zeros((2, n_features))
init_conds[0, 1] = c_true(0) if case == 1 else -c_true(0)
init_conds[1, 0] = c_true(0) if case == 1 else -c_true(0)

def epanechnikov_kernel(u):
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)
def gaussian_kernel(u, sigma=1.0):
    return np.exp(-0.5 * (u / sigma)**2) / (sigma * np.sqrt(2 * np.pi))


print("\n" + "="*60)
print("JOINT OPTIMIZATION")
print("="*60)

tv_optimizer = LassoTimeRegression(
    iterations=2000, 
    l1_penalty=0.001, 
    bandwidth=0.8, 
    kernel=epanechnikov_kernel, 
    fit_intercept=False,
    use_prior=True, 
    tau=1000.0, 
    prior_indices=[0, 0] 
)

model = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=0.01, normalize_columns=False),
    fixed_coefs=fixed_coefs,
    fixed_values=fixed_values,
    time_varying_coefs=time_varying_coefs,
    tv_optimizer=tv_optimizer,
    no_normalization_for_fixeds=True,
    init_conds=init_conds,
    options={'use_selector': False, 'smooth_coefs': True}
)

model.max_iter = 100
model.fit(Theta, x_dot, t=t)

c_est_eq0 = model.tv_coefs_[0][:, 0] 
c_est_eq1 = model.tv_coefs_[1][:, 0]  
c_est = (c_est_eq0 + c_est_eq1) / 2  

const_dx = model.coef_[0, 0]
const_dy = model.coef_[1, 1] 

def reconstruct(t_val, z, t_arr, c_vals):
    idx = np.argmin(np.abs(t_arr - t_val))
    x_val, y_val = z
    c = c_vals[idx]
    if case == 1:
        return [c * y_val, c * x_val]
    else:
        return [-c * y_val, -c * x_val]

sol_rec = solve_ivp(lambda tv, zv: reconstruct(tv, zv, t, c_est),
                    [t[0], t[-1]], [1.0, 0.5], t_eval=t)
x_rec = sol_rec.y.T

c_true_used = c_true(t) if case == 1 else -c_true(t)
r2_c = r2_score(c_true_used, c_est)
mse_c = mean_squared_error(c_true_used, c_est)
mse_traj = mean_squared_error(x_clean, x_rec)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"c(t) (sigmoid):   R² = {r2_c:.4f},   MSE = {mse_c:.2e}")
print(f"Trajectory:       MSE = {mse_traj:.2e}")
print(f"\nConstant coefficients (should be ~0):")
print(f"  dx/dt (x): {const_dx:.6f} (true: {const_dx_true})")
print(f"  dy/dt (y): {const_dy:.6f} (true: {const_dy_true})")
print(f"{'='*60}")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

axes[0, 0].plot(t, c_true_used, 'k-', lw=2, label='True')
axes[0, 0].plot(t, c_est, 'r--', lw=1.5, label='Estimated')
axes[0, 0].set_title(f'$c(t)$ reconstruction\n$R^2 = {r2_c:.4f}$')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('$c(t)$')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, c_true_used, 'k-', lw=2, label='True')
axes[0, 1].plot(t, c_est_eq0, 'b--', lw=1, alpha=0.7, label='From dx/dt')
axes[0, 1].plot(t, c_est_eq1, 'g--', lw=1, alpha=0.7, label='From dy/dt')
axes[0, 1].plot(t, c_est, 'r-', lw=1.5, label='Average')
axes[0, 1].set_title('Estimates from both equations')
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('$c(t)$')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(t, c_true_used - c_est, 'r-', lw=1, alpha=0.7)
axes[0, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 2].fill_between(t, c_true_used - c_est, 0, alpha=0.3, color='red')
axes[0, 2].set_title(f'Reconstruction error\nMSE = {mse_c:.2e}')
axes[0, 2].set_xlabel('t')
axes[0, 2].set_ylabel('Error')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].plot(x_clean[:, 0], x_clean[:, 1], 'k-', lw=2, label='True (clean)')
axes[1, 0].plot(x_noisy[:, 0], x_noisy[:, 1], 'b-', lw=0.5, alpha=0.3, label='Noisy data')
axes[1, 0].plot(x_rec[:, 0], x_rec[:, 1], 'r--', lw=1.5, label='Reconstructed')
axes[1, 0].set_title(f'Phase portrait\nMSE = {mse_traj:.2e}')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axis('equal')

axes[1, 1].plot(t, x_clean[:, 0], 'k-', lw=1.5, alpha=0.8, label='$x(t)$ true')
axes[1, 1].plot(t, x_noisy[:, 0], 'b-', lw=0.5, alpha=0.3, label='$x(t)$ noisy')
axes[1, 1].plot(t, x_rec[:, 0], 'r--', lw=1.2, label='$x(t)$ rec.')
axes[1, 1].plot(t, x_clean[:, 1], 'k-', lw=1.5, alpha=0.8, label='$y(t)$ true')
axes[1, 1].plot(t, x_noisy[:, 1], 'c-', lw=0.5, alpha=0.3, label='$y(t)$ noisy')
axes[1, 1].plot(t, x_rec[:, 1], 'g--', lw=1.2, label='$y(t)$ rec.')
axes[1, 1].set_title('Time series')
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('x, y')
axes[1, 1].legend(loc='upper right', fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

const_true = [0.0, 0.0]
const_est = [const_dx, const_dy]
x_pos = [0, 1]
axes[1, 2].bar(x_pos, const_true, width=0.35, label='True', alpha=0.7, color='gray')
axes[1, 2].bar([p + 0.35 for p in x_pos], const_est, width=0.35, label='Estimated', alpha=0.7, color='steelblue')
axes[1, 2].set_xticks([p + 0.175 for p in x_pos])
axes[1, 2].set_xticklabels(['$\\alpha$ (dx/dt, x)', '$\\beta$ (dy/dt, y)'])
axes[1, 2].set_ylabel('Coefficient value')
axes[1, 2].set_title('Constant coefficients (should be 0)')
axes[1, 2].legend()
axes[1, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('simple_system_results.png', dpi=150, bbox_inches='tight')
print("\nSaved: simple_system_results.png")
plt.show()

print("\n" + "="*60)
print("ADDITIONAL METRICS")
print("="*60)
print(f"Pearson correlation for c(t): {np.corrcoef(c_true_used, c_est)[0,1]:.4f}")
print(f"Mean c(t): true={np.mean(c_true_used):.4f}, estimated={np.mean(c_est):.4f}")
print(f"Std c(t):  true={np.std(c_true_used):.4f}, estimated={np.std(c_est):.4f}")
print(f"SNR: {20*np.log10(np.std(x_clean)/noise_level/np.std(x_clean)):.1f} dB")
print("="*60)