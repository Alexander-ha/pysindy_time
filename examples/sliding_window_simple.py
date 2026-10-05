import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.metrics import r2_score
import pysindy as ps
import warnings
warnings.filterwarnings("ignore")

# 1. Генерация данных: Одномасштабная динамика (Осциллятор)
dt = 0.05
t = np.arange(0, 40.0, dt)

# Плавное изменение частоты (один масштаб времени)
omega_true = lambda t: 1.5 + 0.5 * np.sin(0.2 * t)

def oscillator(t, z):
    x, y = z
    return [-omega_true(t) * y,
             omega_true(t) * x]

sol = solve_ivp(oscillator, [0, 40.0], [1.0, 0.0], t_eval=t)
x_clean = sol.y.T

# Добавляем 2% шума
np.random.seed(42)
noise_level = 0.02
x_noisy = x_clean + noise_level * np.std(x_clean, axis=0) * np.random.randn(*x_clean.shape)

# Робастная производная
sfd = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 15})
dx_smoothed = sfd(x_noisy, t)

# 2. Настройки окна
window_size = 120  # Окно покрывает несколько периодов колебаний
window_shift = 5
ixes = range(0, x_noisy.shape[0] - window_size + 1, window_shift)

# 3. Модель SINDy
feature_lib = ps.PolynomialLibrary(degree=1) 
# Библиотека: [1, x, y]
optimizer = ps.STLSQ(threshold=0.05, alpha=0.01)
base_model = ps.SINDy(feature_library=feature_lib, optimizer=optimizer)

print("Запуск Sliding Window SINDy на гармоническом осцилляторе...")
results = []
for i, start_idx in enumerate(ixes):
    end_idx = start_idx + window_size
    sub_x = x_noisy[start_idx:end_idx]
    sub_t = t[start_idx:end_idx]
    sub_dx = dx_smoothed[start_idx:end_idx]
    
    base_model.fit(sub_x, t=sub_t, x_dot=sub_dx)
    coefs = base_model.coefficients()
    
    # Индексы: 0 -> 1 (константа), 1 -> x, 2 -> y
    results.append({
        't_center': sub_t[window_size // 2],
        'omega_est_1': -coefs[0, 2],  # Из ур-я 0: коэффициент при y (с минусом)
        'omega_est_2': coefs[1, 1],   # Из ур-я 1: коэффициент при x
        'const_x': coefs[0, 1],       # Должен быть 0
        'const_y': coefs[1, 2]        # Должен быть 0
    })

# 4. Оценка результатов
df = pd.DataFrame(results)
t_windows = df['t_center'].values
r2_omega = r2_score(omega_true(t_windows), df['omega_est_1'])

print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ (Одномасштабная динамика)")
print("=" * 50)
print(f"ω(t): R² = {r2_omega:.4f}")
print(f"Ложный признак x (Ур 0): среднее = {df['const_x'].mean():.4f} (истина 0.0)")
print(f"Ложный признак y (Ур 1): среднее = {df['const_y'].mean():.4f} (истина 0.0)")
print("=" * 50 + "\n")

# 5. Визуализация
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Sliding Window на одномасштабной динамике', fontsize=14, fontweight='bold')

axes[0].plot(t, omega_true(t), 'k-', lw=2, label=r'Истина $\omega(t)$')
axes[0].plot(t_windows, df['omega_est_1'], 'r-', lw=2, alpha=0.8, label='Оценка (Окно)')
axes[0].set_title(fr'Динамика параметра $\omega(t)$' + '\n' + f'R²={r2_omega:.3f}')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(t_windows, df['const_x'], 'b-', label='Коэф. при $x$ (Ур 0)')
axes[1].plot(t_windows, df['const_y'], 'g-', label='Коэф. при $y$ (Ур 1)')
axes[1].set_title('Ложные коэффициенты (Дрейф около нуля)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('harmonic_sliding.png', dpi=150)
plt.show()