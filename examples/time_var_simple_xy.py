import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline, interp1d
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
T = 10.0
t = np.arange(0, T, dt)

print(f"Time points: {len(t)}")
print(f"dt = {dt}, T = {T}")

c1_true = lambda t: 1.2 + 0.8 * np.sin(0.6 * t) 
c2_true = lambda t: 0.8 * np.cos(0.8 * t) 
c3_true = lambda t: 0.7 * np.sin(1.0 * t)
def system(t, z):
    x, y = z
    return [c1_true(t) * y + c3_true(t) * x * y,
            c2_true(t) * x]

sol = solve_ivp(system, [0, T], [1.0, 0.5], t_eval=t, method='RK45', rtol=1e-6, atol=1e-8)

if len(sol.t) != len(t):
    x_interp = interp1d(sol.t, sol.y[0, :], kind='linear', fill_value='extrapolate')
    y_interp = interp1d(sol.t, sol.y[1, :], kind='linear', fill_value='extrapolate')
    x_clean = np.column_stack([x_interp(t), y_interp(t)])
else:
    x_clean = sol.y.T

print(f"x_clean shape: {x_clean.shape}")

x_dot = np.zeros_like(x_clean)
for j in range(2):
    spl = UnivariateSpline(t, x_clean[:, j], s=len(t)*1e-4)
    x_dot[:, j] = spl.derivative()(t)

x, y = x_clean[:, 0], x_clean[:, 1]

Theta = np.column_stack([
    y, x, x, y, x*y, x**2, y**2, x**3, y**3,
    np.sin(x), np.cos(x), np.sin(y), np.cos(y)
])

names = ['y_tv', 'x_const', 'x_tv', 'y_const', 'xy_tv', 
         'x²', 'y²', 'x³', 'y³', 'sin(x)', 'cos(x)', 'sin(y)', 'cos(y)']
n_features = Theta.shape[1]

print("="*60)
print("DYNAMICAL SYSTEM")
print("="*60)
print("dx/dt = c₁(t)·y + c₃(t)·xy")
print("dy/dt = c₂(t)·x")
print("="*60)

fixed_coefs_dx = np.zeros((1, n_features), dtype=bool)
fixed_values_dx = np.zeros((1, n_features))
time_varying_coefs_dx = np.zeros((1, n_features), dtype=bool)

time_varying_coefs_dx[0, 0] = True
time_varying_coefs_dx[0, 4] = True

for idx in range(n_features):
    if idx not in [0, 4, 1, 5]:
        fixed_coefs_dx[0, idx] = True
        fixed_values_dx[0, idx] = 0

init_conds_dx = np.zeros((1, n_features))
init_conds_dx[0, 0] = c1_true(0)
init_conds_dx[0, 4] = c3_true(0)

fixed_coefs_dy = np.zeros((1, n_features), dtype=bool)
fixed_values_dy = np.zeros((1, n_features))
time_varying_coefs_dy = np.zeros((1, n_features), dtype=bool)

time_varying_coefs_dy[0, 2] = True

for idx in range(n_features):
    if idx not in [2, 3, 6]:
        fixed_coefs_dy[0, idx] = True
        fixed_values_dy[0, idx] = 0

init_conds_dy = np.zeros((1, n_features))
init_conds_dy[0, 2] = c2_true(0)

def epanechnikov_kernel(u):
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

print("\n" + "="*60)
print("TRAINING")
print("="*60)

bw_dx = 0.8
bw_dy = 0.6

print(f"Bandwidths: dx/dt={bw_dx}s ({int(bw_dx/dt)} pts), dy/dt={bw_dy}s ({int(bw_dy/dt)} pts)")

tv_optimizer_dx = LassoTimeRegression(
    iterations=1000, 
    l1_penalty=0.5,
    bandwidth=bw_dx,
    kernel=epanechnikov_kernel, 
    fit_intercept=False,
    use_prior=True, 
    tau=200.0,
    prior_indices=[0, 1]
)

tv_optimizer_dy = LassoTimeRegression(
    iterations=1000, 
    l1_penalty=0.5,
    bandwidth=bw_dy,
    kernel=epanechnikov_kernel, 
    fit_intercept=False,
    use_prior=True, 
    tau=200.0,
    prior_indices=[0]
)

model_dx = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=0.5, normalize_columns=False),
    fixed_coefs=fixed_coefs_dx,
    fixed_values=fixed_values_dx,
    time_varying_coefs=time_varying_coefs_dx,
    tv_optimizer=tv_optimizer_dx,
    no_normalization_for_fixeds=True,
    init_conds=init_conds_dx,
    options={'use_selector': False, 'smooth_coefs': True}
)

model_dy = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=0.5, normalize_columns=False),
    fixed_coefs=fixed_coefs_dy,
    fixed_values=fixed_values_dy,
    time_varying_coefs=time_varying_coefs_dy,
    tv_optimizer=tv_optimizer_dy,
    no_normalization_for_fixeds=True,
    init_conds=init_conds_dy,
    options={'use_selector': False, 'smooth_coefs': True}
)

model_dx.max_iter = 50
model_dy.max_iter = 50
model_dx.fit(Theta, x_dot[:, 0].reshape(-1, 1), t=t)
model_dy.fit(Theta, x_dot[:, 1].reshape(-1, 1), t=t)

c1_est = model_dx.tv_coefs_[0][:, 0]
c3_est = model_dx.tv_coefs_[0][:, 1]
c2_est = model_dy.tv_coefs_[0][:, 0]

const_dx = model_dx.coef_[0, :]
const_dy = model_dy.coef_[0, :]

c1_smooth = gaussian_filter1d(c1_est, sigma=2)
c2_smooth = gaussian_filter1d(c2_est, sigma=2)
c3_smooth = gaussian_filter1d(c3_est, sigma=2)

c1_interp = interp1d(t, c1_smooth, kind='linear', fill_value='extrapolate')
c2_interp = interp1d(t, c2_smooth, kind='linear', fill_value='extrapolate')
c3_interp = interp1d(t, c3_smooth, kind='linear', fill_value='extrapolate')

def reconstruct_ode(t_val, z, c1_f, c2_f, c3_f):
    x_val, y_val = z
    dxdt = c1_f(t_val) * y_val + c3_f(t_val) * x_val * y_val
    dydt = c2_f(t_val) * x_val
    return [dxdt, dydt]

sol_rec = solve_ivp(
    lambda tv, zv: reconstruct_ode(tv, zv, c1_interp, c2_interp, c3_interp),
    [t[0], t[-1]], [1.0, 0.5], t_eval=t, method='RK45', rtol=1e-6, atol=1e-8
)

if len(sol_rec.t) != len(t):
    x_rec_interp = interp1d(sol_rec.t, sol_rec.y[0, :], kind='linear', fill_value='extrapolate')
    y_rec_interp = interp1d(sol_rec.t, sol_rec.y[1, :], kind='linear', fill_value='extrapolate')
    x_rec = np.column_stack([x_rec_interp(t), y_rec_interp(t)])
else:
    x_rec = sol_rec.y.T

r2_c1 = r2_score(c1_true(t), c1_smooth)
r2_c2 = r2_score(c2_true(t), c2_smooth)
r2_c3 = r2_score(c3_true(t), c3_smooth)
mse_traj = mean_squared_error(x_clean, x_rec)

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
print(f"c₁(t):        R² = {r2_c1:.4f}")
print(f"c₂(t):        R² = {r2_c2:.4f}")
print(f"c₃(t):        R² = {r2_c3:.4f}")
print(f"Trajectory MSE: {mse_traj:.2e}")
print(f"\n{'='*60}")
print("SPARSITY DISCOVERY RESULTS")
print(f"{'='*60}")

test_dx_indices = [1, 5]
print(f"\n📊 dx/dt - testing (should be ZERO):")
correct_zero_dx = 0
for idx in test_dx_indices:
    is_zero = abs(const_dx[idx]) < 1e-6
    status = "✅ ZERO" if is_zero else "❌ NON-ZERO"
    print(f"     {names[idx]}: {const_dx[idx]:.6f} {status}")
    if is_zero:
        correct_zero_dx += 1

test_dy_indices = [3, 6]
print(f"\n📊 dy/dt - testing (should be ZERO):")
correct_zero_dy = 0
for idx in test_dy_indices:
    is_zero = abs(const_dy[idx]) < 1e-6
    status = "✅ ZERO" if is_zero else "❌ NON-ZERO"
    print(f"     {names[idx]}: {const_dy[idx]:.6f} {status}")
    if is_zero:
        correct_zero_dy += 1

total_tested = 4
total_correct = correct_zero_dx + correct_zero_dy

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Sparsity accuracy: {100*total_correct/total_tested:.1f}% ({total_correct}/{total_tested})")
print(f"{'='*60}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].plot(t, c1_true(t), 'k-', lw=2, label='True')
axes[0, 0].plot(t, c1_smooth, 'r--', lw=1.5, label='Estimated')
axes[0, 0].set_title(f'$c_1(t)$\n$R^2 = {r2_c1:.4f}$')
axes[0, 0].set_xlabel('t')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, c2_true(t), 'k-', lw=2, label='True')
axes[0, 1].plot(t, c2_smooth, 'r--', lw=1.5, label='Estimated')
axes[0, 1].set_title(f'$c_2(t)$\n$R^2 = {r2_c2:.4f}$')
axes[0, 1].set_xlabel('t')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(t, c3_true(t), 'k-', lw=2, label='True')
axes[0, 2].plot(t, c3_smooth, 'r--', lw=1.5, label='Estimated')
axes[0, 2].set_title(f'$c_3(t)$\n$R^2 = {r2_c3:.4f}$')
axes[0, 2].set_xlabel('t')
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
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

tested_coeffs = [const_dx[1], const_dx[5], const_dy[3], const_dy[6]]
tested_labels = ['dx/x_const', 'dx/x²', 'dy/y_const', 'dy/y²']
colors = ['green' if abs(c) < 1e-6 else 'red' for c in tested_coeffs]

axes[1, 2].bar(np.arange(4), np.abs(tested_coeffs), alpha=0.7, color=colors)
axes[1, 2].set_xticks(np.arange(4))
axes[1, 2].set_xticklabels(tested_labels, rotation=45, ha='right')
axes[1, 2].set_ylabel('|Coefficient|')
axes[1, 2].set_title(f'Sparsity accuracy: {100*total_correct/total_tested:.0f}%')
axes[1, 2].axhline(y=1e-6, color='orange', linestyle='--', label='Threshold')
axes[1, 2].set_yscale('log')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sparsity_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: sparsity_results.png")
plt.show()

print("\n" + "="*60)
print("VERDICT")
print("="*60)
print(f"c₁(t) R² = {r2_c1:.4f}")
print(f"c₂(t) R² = {r2_c2:.4f}")
print(f"c₃(t) R² = {r2_c3:.4f}")
print(f"Sparsity: {total_correct}/{total_tested} ({100*total_correct/total_tested:.0f}%)")
print("="*60)