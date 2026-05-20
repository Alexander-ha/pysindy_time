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
# MATHIEU EQUATION WITH SPARSITY TEST
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

# Generate clean data
sol = solve_ivp(mathieu_system, [0, t[-1]], [1.0, 0.0], t_eval=t, method='RK45', rtol=1e-8, atol=1e-10)
x_clean = sol.y.T

print(f"Time points: {len(t)}")
print(f"x_clean shape: {x_clean.shape}")

# ============================================================================
# NOISE LEVELS TO TEST
# ============================================================================
noise_levels = [0.001, 0.005, 0.01, 0.025, 0.05]  # 0.1%, 0.5%, 1%, 2.5%, 5%

# Store results
rmse_results = {f'{int(level*100)}%': [] for level in noise_levels}
r2_results = {f'{int(level*100)}%': [] for level in noise_levels}
sparsity_results = {f'{int(level*100)}%': [] for level in noise_levels}

def evaluate_noise_level(noise_level, seed=42):
    """Evaluate model at given noise level"""
    np.random.seed(seed)
    
    # Add noise to states
    x_noisy = x_clean + noise_level * np.std(x_clean, axis=0) * np.random.randn(*x_clean.shape)
    
    # Compute derivatives from noisy data with smoothing
    x_dot = np.zeros_like(x_noisy)
    for j in range(2):
        spl = UnivariateSpline(t, x_noisy[:, j], s=len(t) * noise_level * 0.1)
        x_dot[:, j] = spl.derivative()(t)
    
    x1, x2 = x_noisy[:, 0], x_noisy[:, 1]
    
    # Feature library
    Theta = np.column_stack([x1, x2, x1**2, x2**2])
    n_features = Theta.shape[1]
    
    # Model configuration
    fixed_coefs = np.zeros((2, n_features), dtype=bool)
    fixed_values = np.zeros((2, n_features))
    time_varying_coefs = np.zeros((2, n_features), dtype=bool)
    
    fixed_coefs[0, 1] = True
    fixed_values[0, 1] = 1.0
    time_varying_coefs[1, 0] = True
    
    init_conds = np.zeros((2, n_features))
    init_conds[1, 0] = c_true(0)
    
    def epanechnikov_kernel(u):
        return 0.75 * (1 - u**2) * (np.abs(u) <= 1)
    
    bw_dx2 = 1.0
    tv_optimizer = LassoTimeRegression(
        iterations=2000, l1_penalty=0.05, bandwidth=bw_dx2,
        kernel=epanechnikov_kernel, fit_intercept=False,
        use_prior=True, tau=500.0, prior_indices=[0]
    )
    
    model = FixedCoefficientOptimizer(
        base_optimizer=STLSQ(threshold=0.1, alpha=0.1, normalize_columns=False),
        fixed_coefs=fixed_coefs, fixed_values=fixed_values,
        time_varying_coefs=time_varying_coefs,
        tv_optimizer=tv_optimizer, no_normalization_for_fixeds=True,
        init_conds=init_conds, options={'use_selector': False, 'smooth_coefs': True}
    )
    
    model.max_iter = 100
    model.fit(Theta, x_dot, t=t)
    
    c_est = model.tv_coefs_[1][:, 0]
    coef_dx1 = model.coef_[0, :]
    coef_dx2 = model.coef_[1, :]
    
    # Parameter reconstruction
    from scipy.optimize import curve_fit
    def c_model(t, omega_sq, delta_s_fit, delta_d_fit, theta_fit):
        return -omega_sq * ((1 - delta_s_fit) - delta_d_fit * np.cos(theta_fit * t))
    
    try:
        params, _ = curve_fit(c_model, t, c_est, p0=[omega1**2, delta_s, delta_d, theta], maxfev=5000)
        omega_sq_est, delta_s_est, delta_d_est, theta_est = params
        c_fitted = c_model(t, *params)
        r2_c = r2_score(c_true(t), c_fitted)
    except:
        r2_c = r2_score(c_true(t), c_est)
    
    # Trajectory reconstruction
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
    rmse = np.sqrt(mse_traj)
    
    # Sparsity accuracy
    spurious_indices = [2, 3]  # x1² and x2² in both equations
    correct_zero = 0
    for idx in spurious_indices:
        if abs(coef_dx1[idx]) < 1e-6:
            correct_zero += 1
        if abs(coef_dx2[idx]) < 1e-6:
            correct_zero += 1
    
    return rmse, r2_c, correct_zero, x_rec, x_noisy

# ============================================================================
# RUN EVALUATION FOR ALL NOISE LEVELS
# ============================================================================
print("\n" + "="*60)
print("NOISE LEVEL EVALUATION")
print("="*60)

for noise_level in noise_levels:
    rmse, r2_c, sparsity, _, _ = evaluate_noise_level(noise_level)
    pct = int(noise_level * 100)
    rmse_results[f'{pct}%'].append(rmse)
    r2_results[f'{pct}%'].append(r2_c)
    sparsity_results[f'{pct}%'].append(sparsity)
    print(f"Noise {pct}%: RMSE = {rmse:.4e}, R² = {r2_c:.4f}, Sparsity = {sparsity}/4")

# ============================================================================
# PLOT RMSE vs NOISE LEVEL
# ============================================================================
fig1, ax1 = plt.subplots(figsize=(10, 6))

noise_pcts = [int(l*100) for l in noise_levels]
rmse_means = [np.mean(rmse_results[f'{p}%']) for p in noise_pcts]

ax1.plot(noise_pcts, rmse_means, 'o-', color='steelblue', linewidth=2, markersize=8)
ax1.set_xlabel('Noise Level (%)', fontsize=12)
ax1.set_ylabel('RMSE (Trajectory)', fontsize=12)
ax1.set_title('RMSE vs Noise Level for Mathieu Equation', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

for i, (pct, rmse) in enumerate(zip(noise_pcts, rmse_means)):
    ax1.annotate(f'{rmse:.2e}', (pct, rmse), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('mathieu_rmse_vs_noise.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: mathieu_rmse_vs_noise.png")
plt.show()

# ============================================================================
# DETAILED PLOTS FOR NOISE = 0.5% AND 5%
# ============================================================================
print("\n" + "="*60)
print("DETAILED RECONSTRUCTION FOR 0.5% AND 5% NOISE")
print("="*60)

fig2, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot for 0.5% noise
noise_05 = 0.005
np.random.seed(42)
x_noisy_05 = x_clean + noise_05 * np.std(x_clean, axis=0) * np.random.randn(*x_clean.shape)

x_dot_05 = np.zeros_like(x_noisy_05)
for j in range(2):
    spl = UnivariateSpline(t, x_noisy_05[:, j], s=len(t) * noise_05 * 0.1)
    x_dot_05[:, j] = spl.derivative()(t)

Theta = np.column_stack([x_noisy_05[:, 0], x_noisy_05[:, 1], x_noisy_05[:, 0]**2, x_noisy_05[:, 1]**2])

fixed_coefs = np.zeros((2, 4), dtype=bool)
fixed_values = np.zeros((2, 4))
time_varying_coefs = np.zeros((2, 4), dtype=bool)
fixed_coefs[0, 1] = True
fixed_values[0, 1] = 1.0
time_varying_coefs[1, 0] = True
init_conds = np.zeros((2, 4))
init_conds[1, 0] = c_true(0)

tv_optimizer = LassoTimeRegression(
    iterations=2000, l1_penalty=0.05, bandwidth=1.0,
    kernel=lambda u: 0.75 * (1 - u**2) * (np.abs(u) <= 1),
    fit_intercept=False, use_prior=True, tau=500.0, prior_indices=[0]
)

model = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=0.1, alpha=0.1, normalize_columns=False),
    fixed_coefs=fixed_coefs, fixed_values=fixed_values,
    time_varying_coefs=time_varying_coefs,
    tv_optimizer=tv_optimizer, no_normalization_for_fixeds=True,
    init_conds=init_conds, options={'use_selector': False, 'smooth_coefs': True}
)
model.max_iter = 100
model.fit(Theta, x_dot_05, t=t)

c_est_05 = model.tv_coefs_[1][:, 0]

def reconstruct_ode(t_val, z, t_arr, c_vals):
    idx = np.argmin(np.abs(t_arr - t_val))
    x1_val, x2_val = z
    return [x2_val, c_vals[idx] * x1_val]

sol_rec = solve_ivp(lambda tv, zv: reconstruct_ode(tv, zv, t, c_est_05), [t[0], t[-1]], [1.0, 0.0], t_eval=t)
x_rec_05 = sol_rec.y.T

# Plot for 5% noise
noise_05 = 0.05
np.random.seed(42)
x_noisy_5 = x_clean + noise_05 * np.std(x_clean, axis=0) * np.random.randn(*x_clean.shape)

x_dot_5 = np.zeros_like(x_noisy_5)
for j in range(2):
    spl = UnivariateSpline(t, x_noisy_5[:, j], s=len(t) * noise_05 * 0.1)
    x_dot_5[:, j] = spl.derivative()(t)

Theta = np.column_stack([x_noisy_5[:, 0], x_noisy_5[:, 1], x_noisy_5[:, 0]**2, x_noisy_5[:, 1]**2])

model = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=0.1, alpha=0.1, normalize_columns=False),
    fixed_coefs=fixed_coefs, fixed_values=fixed_values,
    time_varying_coefs=time_varying_coefs,
    tv_optimizer=tv_optimizer, no_normalization_for_fixeds=True,
    init_conds=init_conds, options={'use_selector': False, 'smooth_coefs': True}
)
model.max_iter = 100
model.fit(Theta, x_dot_5, t=t)

c_est_5 = model.tv_coefs_[1][:, 0]

sol_rec = solve_ivp(lambda tv, zv: reconstruct_ode(tv, zv, t, c_est_5), [t[0], t[-1]], [1.0, 0.0], t_eval=t)
x_rec_5 = sol_rec.y.T

# Plot x1 for 0.5% noise
axes[0, 0].plot(t, x_clean[:, 0], 'k-', lw=2, label='True')
axes[0, 0].plot(t, x_noisy_05[:, 0], 'b-', lw=0.5, alpha=0.5, label='Noisy (0.5%)')
axes[0, 0].plot(t, x_rec_05[:, 0], 'r--', lw=1.5, label='Reconstructed')
axes[0, 0].set_title('$x_1$ reconstruction (0.5% noise)')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('$x_1$')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot x2 for 0.5% noise
axes[0, 1].plot(t, x_clean[:, 1], 'k-', lw=2, label='True')
axes[0, 1].plot(t, x_noisy_05[:, 1], 'b-', lw=0.5, alpha=0.5, label='Noisy (0.5%)')
axes[0, 1].plot(t, x_rec_05[:, 1], 'r--', lw=1.5, label='Reconstructed')
axes[0, 1].set_title('$x_2$ reconstruction (0.5% noise)')
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('$x_2$')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Phase portrait for 0.5% noise
axes[0, 2].plot(x_clean[:, 0], x_clean[:, 1], 'k-', lw=2, label='True')
axes[0, 2].plot(x_rec_05[:, 0], x_rec_05[:, 1], 'r--', lw=1.5, label='Reconstructed')
axes[0, 2].set_title(f'Phase portrait (0.5% noise)\nRMSE = {np.sqrt(mean_squared_error(x_clean, x_rec_05)):.2e}')
axes[0, 2].set_xlabel('$x_1$')
axes[0, 2].set_ylabel('$x_2$')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].axis('equal')

# Plot x1 for 5% noise
axes[1, 0].plot(t, x_clean[:, 0], 'k-', lw=2, label='True')
axes[1, 0].plot(t, x_noisy_5[:, 0], 'b-', lw=0.5, alpha=0.5, label='Noisy (5%)')
axes[1, 0].plot(t, x_rec_5[:, 0], 'r--', lw=1.5, label='Reconstructed')
axes[1, 0].set_title('$x_1$ reconstruction (5% noise)')
axes[1, 0].set_xlabel('t')
axes[1, 0].set_ylabel('$x_1$')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot x2 for 5% noise
axes[1, 1].plot(t, x_clean[:, 1], 'k-', lw=2, label='True')
axes[1, 1].plot(t, x_noisy_5[:, 1], 'b-', lw=0.5, alpha=0.5, label='Noisy (5%)')
axes[1, 1].plot(t, x_rec_5[:, 1], 'r--', lw=1.5, label='Reconstructed')
axes[1, 1].set_title('$x_2$ reconstruction (5% noise)')
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('$x_2$')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Phase portrait for 5% noise
axes[1, 2].plot(x_clean[:, 0], x_clean[:, 1], 'k-', lw=2, label='True')
axes[1, 2].plot(x_rec_5[:, 0], x_rec_5[:, 1], 'r--', lw=1.5, label='Reconstructed')
axes[1, 2].set_title(f'Phase portrait (5% noise)\nRMSE = {np.sqrt(mean_squared_error(x_clean, x_rec_5)):.2e}')
axes[1, 2].set_xlabel('$x_1$')
axes[1, 2].set_ylabel('$x_2$')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)
axes[1, 2].axis('equal')

plt.tight_layout()
plt.savefig('mathieu_noise_reconstruction.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: mathieu_noise_reconstruction.png")
plt.show()

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Noise':<10} {'RMSE':<12} {'R² c(t)':<12} {'Sparsity':<10}")
print("-" * 50)
for pct in noise_pcts:
    rmse = np.mean(rmse_results[f'{pct}%'])
    r2 = np.mean(r2_results[f'{pct}%'])
    sp = np.mean(sparsity_results[f'{pct}%'])
    print(f"{pct}%:{' '*5} {rmse:.2e}{' '*4} {r2:.4f}{' '*4} {sp}/4")
print("="*60)