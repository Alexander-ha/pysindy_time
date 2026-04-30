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

# Параметры времени
dt = 0.005
t = np.arange(0, 10.0, dt)

# Истинные коэффициенты
c_true = lambda t: 1 / (1 + np.exp(-(t - 5)))
c1_true = lambda t: np.sin(0.8 * t)

# Система с временными коэффициентами
def simple_system(t, z):
    x, y = z
    return [c1_true(t) * y, c_true(t) * x]

# Генерация данных
sol = solve_ivp(simple_system, [0, 10.0], [1.0, 0.5], t_eval=t)
x_clean = sol.y.T

# Вычисление производных (без шума)
x_smooth = x_clean
x_dot_smooth = np.zeros_like(x_clean)
for j in range(2):
    spl = UnivariateSpline(t, x_clean[:, j], s=0)
    x_dot_smooth[:, j] = spl.derivative()(t)

x, y = x_smooth[:, 0], x_smooth[:, 1]

# Упрощённый словарь - только линейные члены
Theta = np.column_stack([x, y])
names = ['x', 'y']
n_features = Theta.shape[1]

# Маски для коэффициентов
fixed_coefs = np.zeros((2, n_features), dtype=bool)
fixed_values = np.zeros((2, n_features))
time_varying_coefs = np.zeros((2, n_features), dtype=bool)

# Настройка структуры системы
time_varying_coefs[0, 1] = True   # c1(t) перед y в уравнении для x
time_varying_coefs[1, 0] = True   # c(t) перед x в уравнении для y

# Начальные условия для TV-коэффициентов
init_conds = np.zeros((2, n_features))
init_conds[0, 1] = c1_true(0)   # sin(0) = 0
init_conds[1, 0] = c_true(0)    # 1/(1+exp(5)) ≈ 0.0067

# Гауссово ядро
def gaussian_kernel(u):
    return np.exp(-u**2 / 2) / np.sqrt(2 * np.pi)

kernel = gaussian_kernel

# ВАЖНО: Сначала создаём TV оптимизатор с правильными параметрами
tv_optimizer = LassoTimeRegression(
    iterations=3000,
    l1_penalty=0.0,
    bandwidth=[0.3, 0.8],      # [для c1(t), для c(t)]
    kernel=kernel,
    fit_intercept=False,
    use_prior=True,
    tau=100.0,
    prior_indices=[0]
)

# Устанавливаем t_values для TV оптимизатора ДО начала работы
# Это важно для ICI метода
tv_optimizer.t_values_ = t.copy()

# Оптимизатор с отключённым автоматическим выбором для первого запуска
# (чтобы избежать ошибки с t_values_)
model = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=1e-8, normalize_columns=False),
    fixed_coefs=fixed_coefs,
    fixed_values=fixed_values,
    time_varying_coefs=time_varying_coefs,
    tv_optimizer=tv_optimizer,
    no_normalization_for_fixeds=True,
    auto_preprocess=False,
    init_conds=init_conds,
    options={
        'use_selector': False,   # Сначала отключаем селектор
        'selector_method': 'ICI',
        'use_time_meanICI': False,
        'smooth_coefs': True,
        'hmin': [0.08, 0.2],
        'hmax': [0.5, 1.5],
        'thresholdICI': 2.5,
        'bootstrap': 40
    }
)

# Патч для совместимости
original_fit = model.fit
def patched_fit(x_, y, t, sample_weight=None, **reduce_kws):
    return original_fit(x_, y, t, sample_weight, **reduce_kws)

model.max_iter = 500
model.fit = patched_fit

# Обучение
print("Начало обучения...")
model.fit(Theta, x_dot_smooth, t=t)
print("Обучение завершено")

# Извлечение оценок
tv_coefs = [model.tv_coefs_[k] for k in range(2)]
c_est = tv_coefs[1][:, 0]     # для уравнения dy/dt
c1_est = tv_coefs[0][:, 0]    # для уравнения dx/dt

# Метрики качества
r2_c = r2_score(c_true(t), c_est)
r2_c1 = r2_score(c1_true(t), c1_est)
mse_c = np.mean((c_true(t) - c_est)**2)
mse_c1 = np.mean((c1_true(t) - c1_est)**2)

print(f"\n{'='*50}")
print(f"РЕЗУЛЬТАТЫ ВОССТАНОВЛЕНИЯ:")
print(f"{'='*50}")
print(f"c(t) (сигмоида):  R² = {r2_c:.4f},  MSE = {mse_c:.2e}")
print(f"c1(t) (синус):    R² = {r2_c1:.4f}, MSE = {mse_c1:.2e}")
print(f"\nКонстантные коэффициенты:")
print(model.coef_)

# Реконструкция траектории
def system_rec(t_val, z, t_arr, c_p, c1_p):
    idx = min(np.argmin(np.abs(t_arr - t_val)), len(t_arr)-1)
    return [c1_p[idx] * z[1], c_p[idx] * z[0]]

sol_rec = solve_ivp(lambda tv, zv: system_rec(tv, zv, t, c_est, c1_est),
                    [t[0], t[-1]], [1.0, 0.5], t_eval=t, method='RK45')
x_rec = sol_rec.y.T
mse_traj = np.mean((x_clean - x_rec)**2)
print(f"\nОшибка реконструкции траектории: MSE = {mse_traj:.2e}")

# Визуализация
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# График c(t)
axes[0, 0].plot(t, c_true(t), 'k-', lw=2.5, label='Истинный')
axes[0, 0].plot(t, c_est, 'r--', lw=1.5, label='Оценённый', alpha=0.8)
axes[0, 0].set_title(f'c(t) = 1/(1+e⁻⁽ᵗ⁻⁵⁾)\nR² = {r2_c:.3f}, MSE = {mse_c:.2e}')
axes[0, 0].set_xlabel('t')
axes[0, 0].set_ylabel('c(t)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# График c1(t)
axes[0, 1].plot(t, c1_true(t), 'k-', lw=2.5, label='Истинный')
axes[0, 1].plot(t, c1_est, 'r--', lw=1.5, label='Оценённый', alpha=0.8)
axes[0, 1].set_title(f'c₁(t) = sin(0.8t)\nR² = {r2_c1:.3f}, MSE = {mse_c1:.2e}')
axes[0, 1].set_xlabel('t')
axes[0, 1].set_ylabel('c₁(t)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Ошибка восстановления c1(t)
axes[0, 2].plot(t, c1_true(t) - c1_est, 'b-', lw=1, alpha=0.7)
axes[0, 2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
axes[0, 2].fill_between(t, c1_true(t) - c1_est, 0, alpha=0.3)
axes[0, 2].set_title(f'Ошибка оценки c₁(t)\nСр.кв. = {mse_c1:.2e}')
axes[0, 2].set_xlabel('t')
axes[0, 2].set_ylabel('Δc₁(t)')
axes[0, 2].grid(True, alpha=0.3)

# Фазовый портрет
axes[1, 0].plot(x_clean[:, 0], x_clean[:, 1], 'k-', lw=2, alpha=0.7, label='Истинная траектория')
axes[1, 0].plot(x_rec[:, 0], x_rec[:, 1], 'r--', lw=1.5, alpha=0.8, label='Реконструкция')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title(f'Фазовый портрет\nMSE = {mse_traj:.2e}')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Временные ряды x(t) и y(t)
axes[1, 1].plot(t, x_clean[:, 0], 'k-', lw=2, alpha=0.7, label='x(t) истинный')
axes[1, 1].plot(t, x_rec[:, 0], 'r--', lw=1.5, alpha=0.8, label='x(t) реконстр.')
axes[1, 1].plot(t, x_clean[:, 1], 'b-', lw=2, alpha=0.7, label='y(t) истинный')
axes[1, 1].plot(t, x_rec[:, 1], 'g--', lw=1.5, alpha=0.8, label='y(t) реконстр.')
axes[1, 1].set_xlabel('t')
axes[1, 1].set_ylabel('x, y')
axes[1, 1].set_title('Временные ряды')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Относительная ошибка по времени
rel_error_x = np.abs(x_clean[:, 0] - x_rec[:, 0]) / (np.abs(x_clean[:, 0]) + 1e-8)
rel_error_y = np.abs(x_clean[:, 1] - x_rec[:, 1]) / (np.abs(x_clean[:, 1]) + 1e-8)
axes[1, 2].semilogy(t, rel_error_x, 'r-', lw=1, alpha=0.7, label='x(t)')
axes[1, 2].semilogy(t, rel_error_y, 'b-', lw=1, alpha=0.7, label='y(t)')
axes[1, 2].set_xlabel('t')
axes[1, 2].set_ylabel('Относительная ошибка')
axes[1, 2].set_title('Относительная ошибка реконструкции')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simple_system_result_fixed.png', dpi=150)
plt.show()

# Дополнительный анализ
print(f"\n{'='*50}")
print(f"ДОПОЛНИТЕЛЬНЫЙ АНАЛИЗ:")
print(f"{'='*50}")
print(f"Среднее значение c(t): истинное = {np.mean(c_true(t)):.4f}, оценённое = {np.mean(c_est):.4f}")
print(f"Среднее значение c1(t): истинное = {np.mean(c1_true(t)):.4f}, оценённое = {np.mean(c1_est):.4f}")
print(f"Стандартное отклонение c1(t): истинное = {np.std(c1_true(t)):.4f}, оценённое = {np.std(c1_est):.4f}")

# Проверка корреляции
corr_c1 = np.corrcoef(c1_true(t), c1_est)[0, 1]
print(f"Корреляция Пирсона для c1(t): {corr_c1:.4f}")

# Проверка максимальных ошибок
max_error_c1 = np.max(np.abs(c1_true(t) - c1_est))
print(f"Максимальная абсолютная ошибка для c1(t): {max_error_c1:.4f}")