import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.signal import savgol_filter, medfilt
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# Пути для внешнего модуля
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импорты из внешнего модуля
from pysindy.optimizers.fixed_base import FixedCoefficientOptimizer
from pysindy.SINDY_timevar.regressors.time_regressor import LassoTimeRegression
from pysindy.optimizers import STLSQ
import pysindy as ps

# ============================================================================
# 1. ГЕНЕРАЦИЯ ДАННЫХ
# ============================================================================
dt = 0.005
t = np.arange(0, 10.0, dt)

c_true = lambda t: 1 / (1 + np.exp(-(t - 5)))

print("Case: dx/dt = c(t)*y, dy/dt = c(t)*x")
def system(t, z):
    x, y = z
    c = c_true(t)
    return [c * y, c * x]

sol = solve_ivp(system, [0, 10.0], [1.0, 0.5], t_eval=t)
x_clean = sol.y.T

np.random.seed(42)
noise_level = 0.01
x_noisy = x_clean + noise_level * np.std(x_clean, axis=0) * np.random.randn(*x_clean.shape)

sfd = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 15})
dx_smoothed = sfd(x_noisy, t)

x_dot = np.zeros_like(x_clean)
for j in range(2):
    spl = UnivariateSpline(t, x_clean[:, j], s=0)
    x_dot[:, j] = spl.derivative()(t)

x, y = x_noisy[:, 0], x_noisy[:, 1]

print("=" * 70)
print("СРАВНЕНИЕ: SLIDING WINDOW vs LOCALLY WEIGHTED SINDy")
print("Система: dx/dt = c(t)*y, dy/dt = c(t)*x")
print("c(t) = 1/(1 + exp(-(t-5)))")
print("=" * 70)

# ============================================================================
# 2. SLIDING WINDOW SINDy
# ============================================================================
print("\n[1] Sliding Window SINDy...")

window_size = 1700
window_shift = 20
ixes = range(0, x_noisy.shape[0] - window_size + 1, window_shift)

feature_lib = ps.PolynomialLibrary(degree=1)
optimizer = ps.STLSQ(threshold=0.01, alpha=0.01)
model = ps.SINDy(feature_library=feature_lib, optimizer=optimizer)

sw_results = []
for start_idx in ixes:
    end_idx = start_idx + window_size
    sub_x = x_noisy[start_idx:end_idx]
    sub_t = t[start_idx:end_idx]
    sub_dx = dx_smoothed[start_idx:end_idx]
    
    try:
        model.fit(sub_x, t=sub_t, x_dot=sub_dx)
        coefs = model.coefficients()
        
        c0 = coefs[0, 2]
        c1 = coefs[1, 1]
        const_x = coefs[0, 1]
        const_y = coefs[1, 2]
        
        if 0 < c0 < 2 and 0 < c1 < 2:
            sw_results.append({
                't_center': sub_t[window_size // 2],
                'c0': c0,
                'c1': c1,
                'c_avg': (c0 + c1) / 2,
                'const_x': const_x,
                'const_y': const_y
            })
    except:
        continue

df_sw = pd.DataFrame(sw_results)
t_sw = df_sw['t_center'].values
c_sw = df_sw['c_avg'].values

# Сглаживание
def smooth_series(t_values, y_values, window=31, poly=3):
    if len(t_values) < 10:
        return t_values, y_values
    
    median = np.median(y_values)
    mad = np.median(np.abs(y_values - median))
    mask = np.abs(y_values - median) < 3 * mad
    t_clean = t_values[mask]
    y_clean = y_values[mask]
    
    if len(t_clean) < 10:
        return t_values, y_values
    
    try:
        f = interp1d(t_clean, y_clean, kind='linear', fill_value='extrapolate')
        y_interp = f(t_values)
        window = min(window, len(y_interp)-1)
        if window % 2 == 0:
            window -= 1
        if window > 3:
            y_smooth = savgol_filter(y_interp, window, poly)
        else:
            y_smooth = y_interp
        return t_values, y_smooth
    except:
        return t_values, y_values

_, c_sw_smooth = smooth_series(t_sw, c_sw, window=31, poly=3)

f_c_sw = interp1d(t_sw, c_sw_smooth, kind='cubic', fill_value='extrapolate')
c_sw_final = f_c_sw(t)
c_sw_final = np.clip(c_sw_final, 0.0, 1.0)

const_x_sw = np.mean(df_sw['const_x'].values)
const_y_sw = np.mean(df_sw['const_y'].values)

# ============================================================================
# 3. LOCALLY WEIGHTED SINDy (ВНЕШНИЙ МОДУЛЬ)
# ============================================================================
print("\n[2] Locally Weighted SINDy (внешний модуль)...")

Theta = np.column_stack([x, y])
n_features = Theta.shape[1]
print(f"  Размер библиотеки: {n_features}")

fixed_coefs = np.zeros((2, n_features), dtype=bool)
fixed_values = np.zeros((2, n_features))
time_varying_coefs = np.zeros((2, n_features), dtype=bool)

time_varying_coefs[0, 1] = True
time_varying_coefs[1, 0] = True

for i in range(n_features):
    if not time_varying_coefs[0, i]:
        fixed_coefs[0, i] = True
        fixed_values[0, i] = 0
    if not time_varying_coefs[1, i]:
        fixed_coefs[1, i] = True
        fixed_values[1, i] = 0

init_conds = np.zeros((2, n_features))
init_conds[0, 1] = c_true(0)
init_conds[1, 0] = c_true(0)

def epanechnikov_kernel(u):
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

tv_optimizer = LassoTimeRegression(
    iterations=2000,
    l1_penalty=0.0,
    bandwidth=0.3,
    kernel=epanechnikov_kernel,
    fit_intercept=False,
    use_prior=True,
    tau=1000.0,
    prior_indices=[0, 0]
)

model_lw = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=0.01, normalize_columns=False),
    fixed_coefs=fixed_coefs,
    fixed_values=fixed_values,
    time_varying_coefs=time_varying_coefs,
    tv_optimizer=tv_optimizer,
    no_normalization_for_fixeds=True,
    init_conds=init_conds,
    options={'use_selector': False, 'smooth_coefs': True}
)

model_lw.max_iter = 80
model_lw.fit(Theta, x_dot, t=t)

c_lw_eq0 = model_lw.tv_coefs_[0][:, 0]
c_lw_eq1 = model_lw.tv_coefs_[1][:, 0]
c_lw = (c_lw_eq0 + c_lw_eq1) / 2

const_x_lw = model_lw.coef_[0, 0]
const_y_lw = model_lw.coef_[1, 1]

# ============================================================================
# 4. ОЦЕНКА КАЧЕСТВА
# ============================================================================
r2_c_sw = r2_score(c_true(t), c_sw_final)
r2_c_lw = r2_score(c_true(t), c_lw)

mse_c_sw = mean_squared_error(c_true(t), c_sw_final)
mse_c_lw = mean_squared_error(c_true(t), c_lw)

def reconstruct(t_eval, c_vals):
    def rhs(t_val, z):
        x_val, y_val = z
        idx = np.argmin(np.abs(t_eval - t_val))
        c = c_vals[idx]
        return [c * y_val, c * x_val]
    sol_rec = solve_ivp(rhs, [t_eval[0], t_eval[-1]], [1.0, 0.5], t_eval=t_eval)
    return sol_rec.y.T

x_rec_sw = reconstruct(t, c_sw_final)
x_rec_lw = reconstruct(t, c_lw)

mse_traj_sw = mean_squared_error(x_clean, x_rec_sw)
mse_traj_lw = mean_squared_error(x_clean, x_rec_lw)

# ============================================================================
# 5. ВЫВОД
# ============================================================================
print("\n" + "=" * 70)
print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
print("=" * 70)

print("\n" + "-" * 70)
print("ВОССТАНОВЛЕНИЕ ПАРАМЕТРА c(t)")
print("-" * 70)
print(f"\n{'Метод':<25} {'R²':<12} {'MSE':<15}")
print("-" * 70)
print(f"{'Sliding Window':<25} {r2_c_sw:<12.4f} {mse_c_sw:<15.2e}")
print(f"{'Locally Weighted':<25} {r2_c_lw:<12.4f} {mse_c_lw:<15.2e}")
print("-" * 70)

print("\n" + "-" * 70)
print("КАЧЕСТВО РЕКОНСТРУКЦИИ ТРАЕКТОРИЙ")
print("-" * 70)
print(f"\n{'Метод':<25} {'MSE траектории':<15}")
print("-" * 70)
print(f"{'Sliding Window':<25} {mse_traj_sw:<15.2e}")
print(f"{'Locally Weighted':<25} {mse_traj_lw:<15.2e}")
print("-" * 70)

print("\n" + "-" * 70)
print("КОНСТАНТНЫЕ КОЭФФИЦИЕНТЫ (должны быть 0)")
print("-" * 70)
print(f"\n{'coef':<20} {'Sliding Window':<20} {'Locally Weighted':<20}")
print("-" * 70)
print(f"{'const_x (dx/dt, x)':<20} {const_x_sw:<20.4f} {const_x_lw:<20.4f}")
print(f"{'const_y (dy/dt, y)':<20} {const_y_sw:<20.4f} {const_y_lw:<20.4f}")
print("-" * 70)

# ============================================================================
# 6. ВИЗУАЛИЗАЦИЯ
# ============================================================================
fig = plt.figure(figsize=(16, 12))
fig.suptitle('Sliding Window vs Locally Weighted SINDy', fontsize=16, fontweight='bold')

ax1 = plt.subplot(2, 3, 1)
ax1.plot(t, c_true(t), 'k-', lw=2, label='Ground truth')
ax1.plot(t, c_sw_final, 'r--', lw=2, label=f'SW (R²={r2_c_sw:.3f})')
ax1.plot(t, c_lw, 'b--', lw=2, label=f'LW (R²={r2_c_lw:.3f})')
ax1.set_title('recovery c(t)')
ax1.set_xlabel('t')
ax1.set_ylabel('c(t)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 3, 2)
ax2.plot(t, c_true(t), 'k-', lw=2, label='Ground truth')
ax2.plot(t_sw, df_sw['c0'].values, 'r.', alpha=0.3, markersize=2, label='SW из dx/dt')
ax2.plot(t_sw, df_sw['c1'].values, 'g.', alpha=0.3, markersize=2, label='SW из dy/dt')
ax2.plot(t, c_sw_final, 'r-', lw=2, label='SW smoothed')
ax2.set_title('SW estimations')
ax2.set_xlabel('t')
ax2.set_ylabel('c(t)')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = plt.subplot(2, 3, 3)
ax3.plot(t, c_true(t) - c_sw_final, 'r-', lw=1.5, alpha=0.7, label='SW')
ax3.plot(t, c_true(t) - c_lw, 'b-', lw=1.5, alpha=0.7, label='LW')
ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax3.set_title('error of recovery c(t)')
ax3.set_xlabel('t')
ax3.set_ylabel('error')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(2, 3, 4)
ax4.plot(x_clean[:, 0], x_clean[:, 1], 'k-', lw=2, label='ground truth')
ax4.plot(x_rec_sw[:, 0], x_rec_sw[:, 1], 'r--', lw=1.5, alpha=0.7,
         label=f'SW (MSE={mse_traj_sw:.2e})')
ax4.plot(x_rec_lw[:, 0], x_rec_lw[:, 1], 'b--', lw=1.5, alpha=0.7,
         label=f'LW (MSE={mse_traj_lw:.2e})')
ax4.set_title('phase portrait')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.axis('equal')

ax5 = plt.subplot(2, 3, 5)
ax5.plot(t, x_clean[:, 0], 'k-', lw=1.5, alpha=0.6, label='x true')
ax5.plot(t, x_rec_sw[:, 0], 'r--', lw=1, alpha=0.7, label='x SW')
ax5.plot(t, x_rec_lw[:, 0], 'b--', lw=1, alpha=0.7, label='x LW')
ax5.set_title('time series (x)')
ax5.set_xlabel('t')
ax5.set_ylabel('x')
ax5.legend()
ax5.grid(True, alpha=0.3)

ax6 = plt.subplot(2, 3, 6)
x_pos = np.arange(2)
width = 0.25
ax6.bar(x_pos - width, [0, 0], width, label='ground truth', color='gray', alpha=0.7)
ax6.bar(x_pos, [const_x_sw, const_y_sw], width, label='SW', color='red', alpha=0.7)
ax6.bar(x_pos + width, [const_x_lw, const_y_lw], width, label='LW', color='blue', alpha=0.7)
ax6.set_xticks(x_pos)
ax6.set_xticklabels(['const_x (dx/dt, x)', 'const_y (dy/dt, y)'])
ax6.set_ylabel('coef val')
ax6.set_title('const coef.')
ax6.legend()
ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('sw_vs_lw_simple_system.png', dpi=150, bbox_inches='tight')
print("\nСохранено: sw_vs_lw_simple_system.png")

print("\n" + "=" * 70)
print("ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ")
print("=" * 70)

print(f"\nКорреляция Пирсона для c(t):")
print(f"  Sliding Window:   {np.corrcoef(c_true(t), c_sw_final)[0,1]:.4f}")
print(f"  Locally Weighted: {np.corrcoef(c_true(t), c_lw)[0,1]:.4f}")

print(f"\nСреднее значение c(t):")
print(f"  Истина:   {np.mean(c_true(t)):.4f}")
print(f"  SW:       {np.mean(c_sw_final):.4f}")
print(f"  LW:       {np.mean(c_lw):.4f}")

print(f"\nСтандартное отклонение c(t):")
print(f"  Истина:   {np.std(c_true(t)):.4f}")
print(f"  SW:       {np.std(c_sw_final):.4f}")
print(f"  LW:       {np.std(c_lw):.4f}")

print(f"\nMAE для c(t):")
print(f"  SW: {np.mean(np.abs(c_true(t) - c_sw_final)):.4f}")
print(f"  LW: {np.mean(np.abs(c_true(t) - c_lw)):.4f}")

print("\n" + "=" * 70)
print(f"ПАРАМЕТРЫ:")
print(f"  Sliding Window: window_size={window_size}, shift={window_shift}")
print(f"  Locally Weighted: bandwidth=0.3, l1_penalty=0.0")
print("=" * 70)