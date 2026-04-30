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

dt = 0.005
t = np.arange(0, 10.0, dt)

c_true = lambda t: 1 / (1 + np.exp(-(t - 5)))
c1_true = lambda t: np.sin(0.8 * t)

def simple_system(t, z):
    x, y = z
    return [c1_true(t) * y, c_true(t) * x]

sol = solve_ivp(simple_system, [0, 10.0], [1.0, 0.5], t_eval=t)
x_clean = sol.y.T

x_smooth = x_clean
x_dot_smooth = np.zeros_like(x_clean)
for j in range(2):
    spl = UnivariateSpline(t, x_clean[:, j], s=0)
    x_dot_smooth[:, j] = spl.derivative()(t)

x, y = x_smooth[:, 0], x_smooth[:, 1]
Theta = np.column_stack([x, y, x**2, y**2, x*y])
names = ['x', 'y', 'x²', 'y²', 'xy']
n_features = Theta.shape[1]

fixed_coefs = np.zeros((2, n_features), dtype=bool)
fixed_values = np.zeros((2, n_features))
time_varying_coefs = np.zeros((2, n_features), dtype=bool)

time_varying_coefs[0, 1] = True   
time_varying_coefs[1, 0] = True   
fixed_coefs[1, 1] = True
fixed_coefs[0,0] = True
fixed_values[0,0] = 0
fixed_values[1, 1] = 0

init_conds = np.zeros((2, n_features))
init_conds[0, 1] = 0.0           
init_conds[1, 0] = 0.0067    

kernel = lambda u: 0.75 * (1 - u**2) * (np.abs(u) <= 1)

model = FixedCoefficientOptimizer(
    base_optimizer=STLSQ(threshold=1e-8, normalize_columns=False),
    fixed_coefs=fixed_coefs,
    fixed_values=fixed_values,
    time_varying_coefs=time_varying_coefs,
    tv_optimizer=LassoTimeRegression(
        iterations=2000, l1_penalty=0.0, bandwidth=[0.4755, 0.9244],
        kernel=kernel, fit_intercept=False,
        use_prior=True, tau=100.0, prior_indices=[0]
    ),
    no_normalization_for_fixeds=True,
    auto_preprocess=False,
    init_conds=init_conds,
    options={    'use_selector': False,
    'selector_method': 'ICI',
    'use_time_meanICI': True,
    'smooth_coefs': False,
    'hmin': 0.15,
    'hmax': [0.4, 0.95],
    'thresholdICI': 3.5,
    'bootstrap': 40}
)

original_fit = model.fit
def patched_fit(x_, y, t=None, sample_weight=None, **reduce_kws):
    return original_fit(x_, y, t, sample_weight, **reduce_kws)
model.max_iter = 1000
model.fit = patched_fit
model.fit(Theta, x_dot_smooth, t=t)

tv_coefs = [model.tv_coefs_[k] for k in range(2)]
c_est = tv_coefs[1][:, 0]
c1_est = tv_coefs[0][:, 0]

r2_c = r2_score(c_true(t), c_est)
r2_c1 = r2_score(c1_true(t), c1_est)
print(f"c(t):  R²={r2_c:.4f}")
print(f"c1(t): R²={r2_c1:.4f}")
print(f"Const: {model.coef_}")
print(f"MSE TV: {np.mean((c_true(t) - c_est)**2):.2e}, {np.mean((c1_true(t) - c1_est)**2):.2e}")

def system_rec(t_val, z, t_arr, c_p, c1_p):
    idx = min(np.argmin(np.abs(t_arr - t_val)), len(t_arr)-1)
    return [c1_p[idx] * z[1], c_p[idx] * z[0]]

sol_rec = solve_ivp(lambda tv, zv: system_rec(tv, zv, t, c_est, c1_est),
                    [t[0], t[-1]], [1.0, 0.5], t_eval=t, method='RK45')
x_rec = sol_rec.y.T
mse = np.mean((x_clean - x_rec)**2)
print(f"MSEres: {mse:.2e}")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(t, c_true(t), 'k-', lw=2, label='True')
axes[0].plot(t, c_est, 'r--', lw=1.5, label='Est')
axes[0].set_title(f'c(t), R²={r2_c:.3f}')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(t, c1_true(t), 'k-', lw=2, label='True')
axes[1].plot(t, c1_est, 'r--', lw=1.5, label='Est')
axes[1].set_title(f'c1(t), R²={r2_c1:.3f}')
axes[1].legend(); axes[1].grid(True)

axes[2].plot(x_clean[:, 0], x_clean[:, 1], 'k-', lw=0.8, alpha=0.7, label='True')
axes[2].plot(x_rec[:, 0], x_rec[:, 1], 'r--', lw=1.2, label='Rec')
axes[2].set_xlabel('x'); axes[2].set_ylabel('y')
axes[2].set_title(f'Phase, MSE={mse:.2e}')
axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.savefig('simple_system_result.png', dpi=150)
plt.show()