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
# MATHIEU EQUATION WITH SPARSITY TEST (ONLY 2 SPURIOUS TERMS)
# ============================================================================
# System:
#   dx1/dt = x2 + SPURIOUS TERMS (should become zero)
#   dx2/dt = c(t)·x1 + SPURIOUS TERMS (should become zero)
# ============================================================================

dt = 0.01
t = np.arange(0, 20.0, dt)

omega1 = 2.0
delta_s = 0.3
delta_d = 0.5
theta = 1.5  
c_true = lambda t: -omega1**2 * ((1 - delta_s) - delta_d * np.cos(theta * t))

def mathieu_system(t, z):
    x1, x2 = z
    return [x2, c_true(t) * x1]
sol = solve_ivp(mathieu_system, [0, t[-1]], [1.0, 0.0], t_eval=t, method='RK45', rtol=1e-8, atol=1e-10)
x_clean = sol.y.T

print(f"Time points: {len(t)}")
print(f"x_clean shape: {x_clean.shape}")
x_dot = np.zeros_like(x_clean)
for j in range(2):
    spl = UnivariateSpline(t, x_clean[:, j], s=1e-6)
    x_dot[:, j] = spl.derivative()(t)

x1, x2 = x_clean[:, 0], x_clean[:, 1]

Theta = np.column_stack([
    x1,
    x2,  
    x1**2, 
    x2**2,
])

names = ['x1_tv', 'x2_const', 'x1²', 'x2²']
n_features = Theta.shape[1]

print("="*60)
print("MATHIEU EQUATION WITH SPARSITY TEST")
print("="*60)
print("dx₁/dt = x₂ + α₁·x₁² + α₂·x₂²")
print("dx₂/dt = c(t)·x₁ + β₁·x₁² + β₂·x₂²")
print(f"c(t) = -ω₁²[(1 - δ_s) - δ_d cos(θ t)]")
print(f"ω₁={omega1}, δ_s={delta_s}, δ_d={delta_d}, θ={theta}")
print(f"\nFeature library size: {n_features}")
print("🔬 SPARSITY TEST: x₁² and x₂² in BOTH equations should be ZERO (4 coefficients total)")
print("="*60)

fixed_coefs = np.zeros((2, n_features), dtype=bool)
fixed_values = np.zeros((2, n_features))
time_varying_coefs = np.zeros((2, n_features), dtype=bool)
fixed_coefs[0, 1] = True
fixed_values[0, 1] = 1.0
time_varying_coefs[1, 0] = True
print(f"\nTime-varying coefficients: {np.sum(time_varying_coefs)}")
print(f"  - dx₂/dt: x₁ (c(t))")
print(f"\n🔬 TESTING SPARSITY ON 4 COEFFICIENTS (should become ZERO):")
print(f"  - dx₁/dt: x₁², x₂²")
print(f"  - dx₂/dt: x₁², x₂²")
init_conds = np.zeros((2, n_features))
init_conds[1, 0] = c_true(0)

def epanechnikov_kernel(u):
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)
print("\n" + "="*60)
print("TRAINING")
print("="*60)

bw_dx2 = 1.0
print(f"Bandwidth: dx₂/dt={bw_dx2}s ({int(bw_dx2/dt)} pts)")

tv_optimizer = LassoTimeRegression(
    iterations=2000,
    l1_penalty=0.05,
    bandwidth=bw_dx2,
    kernel=epanechnikov_kernel,
    fit_intercept=False,
    use_prior=True,
    tau=500.0,
    prior_indices=[0]
)

model = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=0.1,alpha=0.1, normalize_columns=False),
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
c_est = model.tv_coefs_[1][:, 0]
coef_dx1 = model.coef_[0, :]
coef_dx2 = model.coef_[1, :]

print(f"\n{'='*60}")
print("SPARSITY DISCOVERY RESULTS")
print(f"{'='*60}")

print(f"\n📊 FIRST EQUATION (dx₁/dt = x₂ + ...):")
print(f"  x₂ (true term): {coef_dx1[1]:.6f} (should be 1.0)")

correct_zero = 0
total_tested = 4

is_zero = abs(coef_dx1[2]) < 1e-6
status = "✅ ZERO" if is_zero else "❌ NON-ZERO"
print(f"  x₁²: {coef_dx1[2]:.6f} {status} (should be 0)")
if is_zero:
    correct_zero += 1

is_zero = abs(coef_dx1[3]) < 1e-6
status = "✅ ZERO" if is_zero else "❌ NON-ZERO"
print(f"  x₂²: {coef_dx1[3]:.6f} {status} (should be 0)")
if is_zero:
    correct_zero += 1

print(f"\n📊 SECOND EQUATION (dx₂/dt = c(t)·x₁ + ...):")
print(f"  x₁ (time-varying, estimated separately)")

is_zero = abs(coef_dx2[2]) < 1e-6
status = "✅ ZERO" if is_zero else "❌ NON-ZERO"
print(f"  x₁²: {coef_dx2[2]:.6f} {status} (should be 0)")
if is_zero:
    correct_zero += 1

is_zero = abs(coef_dx2[3]) < 1e-6
status = "✅ ZERO" if is_zero else "❌ NON-ZERO"
print(f"  x₂²: {coef_dx2[3]:.6f} {status} (should be 0)")
if is_zero:
    correct_zero += 1

from scipy.optimize import curve_fit

def c_model(t, omega_sq, delta_s_fit, delta_d_fit, theta_fit):
    return -omega_sq * ((1 - delta_s_fit) - delta_d_fit * np.cos(theta_fit * t))

try:
    params, _ = curve_fit(c_model, t, c_est, p0=[omega1**2, delta_s, delta_d, theta], maxfev=5000)
    omega_sq_est, delta_s_est, delta_d_est, theta_est = params
    
    print(f"\n{'='*60}")
    print("PARAMETER RECONSTRUCTION")
    print(f"{'='*60}")
    print(f"ω²:      estimated = {omega_sq_est:.4f}, true = {omega1**2:.4f}")
    print(f"δ_s:     estimated = {delta_s_est:.4f}, true = {delta_s:.4f}")
    print(f"δ_d:     estimated = {delta_d_est:.4f}, true = {delta_d:.4f}")
    print(f"θ:       estimated = {theta_est:.4f}, true = {theta:.4f}")
    
    c_fitted = c_model(t, *params)
    r2_c = r2_score(c_true(t), c_fitted)
except Exception as e:
    print(f"Parameter fitting failed: {e}")
    r2_c = r2_score(c_true(t), c_est)
def reconstruct_ode(t_val, z, t_arr, c_vals):
    idx = np.argmin(np.abs(t_arr - t_val))
    x1_val, x2_val = z
    return [x2_val, c_vals[idx] * x1_val]

sol_rec = solve_ivp(
    lambda tv, zv: reconstruct_ode(tv, zv, t, c_est),
    [t[0], t[-1]], [1.0, 0.0], t_eval=t, method='RK45'
)
x_rec = sol_rec.y.T

mse_traj = mean_squared_error(x_clean, x_rec)
print(f"\n{'='*60}")
print("SPARSITY SUMMARY")
print(f"{'='*60}")
print(f"Spurious coefficients tested: {total_tested}")
print(f"Correctly identified as ZERO: {correct_zero}")
print(f"Sparsity accuracy: {100*correct_zero/total_tested:.1f}%")
print(f"{'='*60}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes[0, 0].plot(t, c_true(t), 'k-', lw=2, label='True')
axes[0, 0].plot(t, c_est, 'r--', lw=1.5, label='Estimated')
try:
    axes[0, 0].plot(t, c_fitted, 'b:', lw=1.5, label='Fitted')
    axes[0, 0].legend()
except:
    pass
axes[0, 0].set_title(f'$c(t)$ reconstruction\n$R^2 = {r2_c:.4f}$')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('c(t)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 1].plot(t, c_true(t) - c_est, 'r-', lw=1, alpha=0.7)
axes[0, 1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 1].set_title(f'c(t) error\nMSE = {mean_squared_error(c_true(t), c_est):.2e}')
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('Error')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 2].plot(x_clean[:, 0], x_clean[:, 1], 'k-', lw=2, label='True')
axes[0, 2].plot(x_rec[:, 0], x_rec[:, 1], 'r--', lw=1.5, label='Reconstructed')
axes[0, 2].set_title(f'Phase portrait\nMSE = {mse_traj:.2e}')
axes[0, 2].set_xlabel('x₁')
axes[0, 2].set_ylabel('x₂')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axis('equal')
axes[1, 0].plot(t, x_clean[:, 0], 'k-', lw=2, alpha=0.8, label='$x_1$ true')
axes[1, 0].plot(t, x_rec[:, 0], 'r--', lw=1.5, label='$x_1$ rec.')
axes[1, 0].set_title('Time series $x_1$')
axes[1, 0].set_xlabel('t')
axes[1, 0].set_ylabel('$x_1$')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 1].plot(t, x_clean[:, 1], 'b-', lw=2, alpha=0.8, label='$x_2$ true')
axes[1, 1].plot(t, x_rec[:, 1], 'g--', lw=1.5, label='$x_2$ rec.')
axes[1, 1].set_title('Time series $x_2$')
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('$x_2$')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

tested_names = ['eq1/x₁²', 'eq1/x₂²', 'eq2/x₁²', 'eq2/x₂²']
tested_values = [abs(coef_dx1[2]), abs(coef_dx1[3]), abs(coef_dx2[2]), abs(coef_dx2[3])]
tested_colors = ['green' if v < 1e-6 else 'red' for v in tested_values]

axes[1, 2].bar(np.arange(4), tested_values, alpha=0.7, color=tested_colors)
axes[1, 2].set_xticks(np.arange(4))
axes[1, 2].set_xticklabels(tested_names, rotation=45, ha='right')
axes[1, 2].set_ylabel('|Coefficient|')
axes[1, 2].set_title(f'Sparsity accuracy: {100*correct_zero/total_tested:.0f}%')
axes[1, 2].axhline(y=1e-6, color='orange', linestyle='--', label='Threshold')
axes[1, 2].set_yscale('log')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('mathieu_sparsity_test.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: mathieu_sparsity_test.png")
plt.show()

print("\n" + "="*60)
print("VERDICT")
print("="*60)
print(f"c(t) R² = {r2_c:.4f}")
print(f"Trajectory MSE = {mse_traj:.2e}")
print(f"Sparsity accuracy: {100*correct_zero/total_tested:.1f}% ({correct_zero}/{total_tested})")
print("="*60)