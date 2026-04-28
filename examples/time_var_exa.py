import sys, os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pysindy.optimizers.fixed_base import FixedCoefficientOptimizer
from pysindy.SINDY_timevar.regressors.time_regressor import LassoTimeRegression
from pysindy.optimizers import STLSQ

# ============================================================
# ДАННЫЕ
# ============================================================
dt = 0.01
t = np.arange(0, 5.0, dt)

sigma_true = lambda t: 10 + 2 * np.sin(1.5 * t)
beta_true  = lambda t: 8/3 + 0.5 * np.cos(0.3 * t)

def lorenz(t, z):
    x, y, zz = z
    return [sigma_true(t) * (y - x),
            x * (28 - zz) - y,
            x * y - beta_true(t) * zz]

sol = solve_ivp(lorenz, [0, 5.0], [-8, 7, 27], t_eval=t)
x_clean = sol.y.T

# Сглаживание и дифференцирование
t_spline = np.arange(0, 5.0, dt)
x_smooth = np.zeros_like(x_clean)
x_dot_smooth = np.zeros_like(x_clean)
for j in range(3):
    spl = UnivariateSpline(t_spline, x_clean[:, j], s=0)
    x_smooth[:, j] = spl(t_spline)
    x_dot_smooth[:, j] = spl.derivative()(t)

# ============================================================
# БИБЛИОТЕКА С РАЗНОСТНЫМИ ИНВАРИАНТАМИ
# ============================================================
# Конструируем признаки вручную:
# x, y, z, x², y², z², xy, xz, yz, (y-x), (z-y), (x-z), xyz
x, y, z = x_smooth[:, 0], x_smooth[:, 1], x_smooth[:, 2]

Theta = np.column_stack([
    x, y, z,                    # 0,1,2: линейные
    x**2, y**2, z**2,           # 3,4,5: квадраты
    x*y, x*z, y*z,              # 6,7,8: парные произведения
    y - x,                      # 9: разность (для σ)
    z - y,                      # 10: разность
    x - z,                      # 11: разность
    x*y*z                       # 12: тройное произведение
])

n_features = Theta.shape[1]
names = ['x', 'y', 'z', 'x²', 'y²', 'z²', 'xy', 'xz', 'yz',
         'y-x', 'z-y', 'x-z', 'xyz']
print(f"Библиотека: {names}")
print(f"Форма Theta: {Theta.shape}")

# ============================================================
# МАСКИ (ИСПОЛЬЗУЕМ РАЗНОСТНЫЕ ПРИЗНАКИ)
# ============================================================
kernel = lambda u: 0.75 * (1 - u**2) * (np.abs(u) <= 1)

fixed_coefs = np.zeros((3, n_features), dtype=bool)
fixed_values = np.zeros((3, n_features))
time_varying_coefs = np.zeros((3, n_features), dtype=bool)

# Уравнение 0: dx/dt = σ(t)*(y-x)
# σ(t) — TV, признак 9 (y-x)
time_varying_coefs[0, 9] = True

# Уравнение 1: dy/dt = 28*x - y - x*z
fixed_coefs[1, 0] = True; fixed_values[1, 0] = 28.0   # x
fixed_coefs[1, 1] = True; fixed_values[1, 1] = -1.0   # y
fixed_coefs[1, 7] = True; fixed_values[1, 7] = -1.0   # xz

# Уравнение 2: dz/dt = 1.0*xy - β(t)*z
fixed_coefs[2, 6] = True; fixed_values[2, 6] = 1.0    # xy
time_varying_coefs[2, 2] = True                        # z — TV (β)

# Фиксируем ВСЕ остальные нулями
for eq in range(3):
    for j in range(n_features):
        if not time_varying_coefs[eq, j] and not fixed_coefs[eq, j]:
            fixed_coefs[eq, j] = True
            fixed_values[eq, j] = 0.0

# ============================================================
# PRIOR НА НАЧАЛЬНЫЕ ЗНАЧЕНИЯ
# ============================================================
init_conds = np.zeros((3, n_features))
init_conds[0, 9] = sigma_true(0)    # σ(0) для признака y-x
init_conds[2, 2] = -beta_true(0)    # -β(0) для признака z

# ============================================================
# ОПЦИИ ICI
# ============================================================
ici_options = {
    'use_selector': True,
    'selector_method': 'ICI',
    'use_time_meanICI': True,
    'smooth_coefs': True,
    'hmin': 0.5,
    'hmax': [1.1, 2.0, 1.3],
    'thresholdICI': 3.0,
    'bootstrap': 60
}

# ============================================================
# СОВМЕСТНАЯ ОЦЕНКА
# ============================================================
print("=" * 60)
print("СОВМЕСТНАЯ ОЦЕНКА σ(t) И β(t)")
print("=" * 60)

model = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=1e-8, normalize_columns=False),
    fixed_coefs=fixed_coefs,
    fixed_values=fixed_values,
    time_varying_coefs=time_varying_coefs,
    tv_optimizer=LassoTimeRegression(
        iterations=2000, l1_penalty=0.0, bandwidth=1.3,
        kernel=kernel, use_prior=True, tau=1000.0,
        fit_intercept=False, prior_indices=[0]
    ),
    no_normalization_for_fixeds=True,
    auto_preprocess=False,
    init_conds=init_conds,
    options=ici_options
)
model.fit(Theta, x_dot_smooth, t=t)

# ============================================================
# ИЗВЛЕЧЕНИЕ РЕЗУЛЬТАТОВ
# ============================================================
tv_coefs = [model.tv_coefs_[k] for k in range(3)]

# σ(t) из признака 9 (y-x) в уравнении 0
sigma_est = tv_coefs[0][:, 0]  # единственный TV-признак в уравнении 0

# β(t) из признака 2 (z) в уравнении 2
beta_raw = tv_coefs[2][:, 0]   # это -β(t)
beta_est = -beta_raw            # +β(t)

# ============================================================
# МЕТРИКИ
# ============================================================
r2_sigma = r2_score(sigma_true(t), sigma_est)
r2_beta  = r2_score(beta_true(t), beta_est)
corr_sigma = np.corrcoef(sigma_true(t), sigma_est)[0, 1]
corr_beta  = np.corrcoef(beta_true(t), beta_est)[0, 1]

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"σ(t): R² = {r2_sigma:.4f}, corr = {corr_sigma:.4f}, mean = {np.mean(sigma_est):.2f} (true ≈ 10.8)")
print(f"β(t): R² = {r2_beta:.4f}, corr = {corr_beta:.4f}, mean = {np.mean(beta_est):.2f} (true ≈ 3.0)")

# Проверка фиксированных коэффициентов
print(f"\nФиксированные коэффициенты:")
print(f"  Ур.1: x={model.coef_[1,0]:.1f}, y={model.coef_[1,1]:.1f}, xz={model.coef_[1,7]:.1f}")
print(f"  Ур.2: xy={model.coef_[2,6]:.1f}")

# ============================================================
# РЕКОНСТРУКЦИЯ АТТРАКТОРА
# ============================================================
rho_est = model.coef_[1, 0]

def lorenz_reconstructed(t_val, z, t_arr, s_arr, b_arr, rho):
    idx = np.argmin(np.abs(t_arr - t_val))
    idx = min(idx, len(t_arr) - 1)
    x, y, zz = z
    return [s_arr[idx] * (y - x),
            x * (rho - zz) - y,
            x * y + b_arr[idx] * zz]

sol_rec = solve_ivp(
    lambda tv, zv: lorenz_reconstructed(tv, zv, t, sigma_est, beta_raw, rho_est),
    [t[0], t[-1]], [-8, 7, 27], t_eval=t, method='RK45'
)
x_rec = sol_rec.y.T
mse = np.mean((x_clean - x_rec) ** 2)
print(f"\nMSE реконструкции: {mse:.4f}")
print(f"ρ оценённый: {rho_est:.2f}")

# ============================================================
# ГРАФИКИ
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('FixedCoefficientOptimizer с разностными инвариантами', fontsize=14, fontweight='bold')

# σ(t)
axes[0, 0].plot(t, sigma_true(t), 'k-', lw=2, label='True')
axes[0, 0].plot(t, sigma_est, 'r--', lw=1.5, label='Est')
axes[0, 0].set_title(f'σ(t): R²={r2_sigma:.3f}, corr={corr_sigma:.3f}')
axes[0, 0].legend(); axes[0, 0].grid(True)

# β(t)
axes[0, 1].plot(t, beta_true(t), 'k-', lw=2, label='True')
axes[0, 1].plot(t, beta_est, 'r--', lw=1.5, label='Est')
axes[0, 1].set_title(f'β(t): R²={r2_beta:.3f}, corr={corr_beta:.3f}')
axes[0, 1].legend(); axes[0, 1].grid(True)

# Сводка
axes[0, 2].axis('off')
summary = f"ρ = {rho_est:.2f} (true 28.0)\nMSE = {mse:.2f}"
axes[0, 2].text(0.1, 0.5, summary, fontsize=14, transform=axes[0, 2].transAxes)

# x-y
axes[1, 0].plot(x_clean[:, 0], x_clean[:, 1], 'k-', lw=0.8, alpha=0.7)
axes[1, 0].plot(x_rec[:, 0], x_rec[:, 1], 'r--', lw=1.2)
axes[1, 0].set_xlabel('x'); axes[1, 0].set_ylabel('y')
axes[1, 0].grid(True); axes[1, 0].set_title('x-y')

# x-z
axes[1, 1].plot(x_clean[:, 0], x_clean[:, 2], 'k-', lw=0.8, alpha=0.7)
axes[1, 1].plot(x_rec[:, 0], x_rec[:, 2], 'r--', lw=1.2)
axes[1, 1].set_xlabel('x'); axes[1, 1].set_ylabel('z')
axes[1, 1].grid(True); axes[1, 1].set_title('x-z')

# y-z
axes[1, 2].plot(x_clean[:, 1], x_clean[:, 2], 'k-', lw=0.8, alpha=0.7)
axes[1, 2].plot(x_rec[:, 1], x_rec[:, 2], 'r--', lw=1.2)
axes[1, 2].set_xlabel('y'); axes[1, 2].set_ylabel('z')
axes[1, 2].grid(True); axes[1, 2].set_title('y-z')

plt.tight_layout()
plt.savefig('lorenz_invariants.png', dpi=150)
plt.show()

print("\nГОТОВО.")