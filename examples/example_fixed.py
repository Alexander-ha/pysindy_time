import sys
import os
import numpy as np

# add to sys.path for simplicity 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pysindy.optimizers.sr3 import SR3
from pysindy.optimizers.frols import FROLS
from pysindy.optimizers.fixed_base import FixedCoefficientOptimizer
from pysindy.optimizers.ssr import SSR


# create synth data for regression
np.random.seed(42)
n_samples = 100
n_features = 5

X = np.random.randn(n_samples, n_features)
true_coef = np.array([1.5, 0, -9.0, 0, 0.8])
y = X@true_coef + np.random.randn(n_samples)*0.1

# create a binary mask 
fixmask = np.array([[False, False, True, False, False]])
fixval = np.array([[1.5, 0, -7, 0, 0]]) 

# ex 1 - using optimizer both with fix mask and init vals 
base_optimizer = FROLS(max_iter=10, alpha=0.05, normalize_columns=False)
proxy_optimizer = FixedCoefficientOptimizer(
    base_optimizer=base_optimizer,
    fixed_coefs=fixmask,
    fixed_values=fixval
)

model = proxy_optimizer.fit(X, y)

print("FROLS: true coefs ", true_coef)
print("FROLS: predicted ", model.coef_[0])
unchangarr = [np.isclose(model.coef_[0][i], true_coef[i], atol=1e-10) for i in range(0, fixval.size)]
print("FROLS: remained unchanged ", unchangarr)
difarr = [abs(model.coef_[0][i] - true_coef[i]) for i in range(0, fixval.size)]
print("FROLS: Difference in fixed coefficient:", )

#pred and calc R^2
y_pred = model.predict(X)
ss_res=np.sum((y-y_pred.flatten())**2)
ss_tot = np.sum((y-np.mean(y))**2)
r2=1-ss_res/ss_tot
print("FROLS:R^2 score:", r2)

# Check preds
print("\nFirst few preds vs acutal:")
for i in range(5):
    print(f"sample {i}: Pred={y_pred[i][0]:.4f}, Actual={y[i]:.4f}")

# ex 2 - no coefficients fixed
print("\n--- (no fixed coefficients) ---")
base_optimizer2 = FROLS(max_iter=10, alpha=0.05, normalize_columns=False)
model2 = base_optimizer2.fit(X, y)
print("Frols: Baseline coefficients:", model2.coef_[0])
y_pred2 =model2.predict(X)
ss_res2=np.sum((y-y_pred2.flatten())**2)
r2_2=1-ss_res2/ss_tot
print("frols: Baseline R^2:",r2_2)
#___________________________________________________________________________
#ex 3 using another one optimizer

X = np.random.randn(n_samples, n_features)
true_coef = np.array([1.5, 0, -9.0, 0, 0.8])
y = X@true_coef + np.random.randn(n_samples)*0.1

# create a binary mask 
fixmask = np.array([[False, False, False, False, False]])
fixval = np.array([[1.5, 0, -7, 0, 0]]) 

base_optimizer = SR3(max_iter=20, normalize_columns=False)
proxy_optimizer = FixedCoefficientOptimizer(
    base_optimizer=base_optimizer,
    fixed_coefs=fixmask,
    fixed_values=fixval
)
model=proxy_optimizer.fit(X, y)

print("sr3:true coefs ",true_coef)
print("sr3:predicted ", model.coef_[0])
unchangarr = [np.isclose(model.coef_[0][i], true_coef[i], atol=1e-10) for i in range(0, fixval.size)]
print("sr3:remained unchanged ", unchangarr)
difarr = [abs(model.coef_[0][i]-true_coef[i]) for i in range(0, fixval.size)]
print("sr3:difference in fixed coefficient:", )
y_pred = model.predict(X)
ss_res=np.sum((y-y_pred.flatten())**2)
ss_tot = np.sum((y-np.mean(y))**2)
r2=1-ss_res/ss_tot
print("sr3:R^2 score:", r2)
#_____________________________________________________________________________________________
n_samples= 2000
n_features=3
X = np.random.randn(n_samples, n_features)
true_coef = np.array([1500, -8, 0.7])
y = X@true_coef + np.random.randn(n_samples)*0.01

# create a binary mask 
fixmask = np.array([[True, False, False]])
fixval = np.array([[1300, -8, 0.7]]) 

base_optimizer = SSR(max_iter=20, normalize_columns=True)
proxy_optimizer = FixedCoefficientOptimizer(
    base_optimizer=base_optimizer,
    fixed_coefs=fixmask,
    fixed_values=fixval,
    no_normalization_for_fixeds=True
)
model=proxy_optimizer.fit(X, y)

print("ssr:true coefs ",true_coef)
print("ssr:predicted ", model.coef_[0])
unchangarr = [np.isclose(model.coef_[0][i], true_coef[i], atol=1e-10) for i in range(0, fixval.size)]
print("ssr:remained unchanged ", unchangarr)
difarr = [abs(model.coef_[0][i]-true_coef[i]) for i in range(0, fixval.size)]
print("ssr:difference in fixed coefficient:", )
y_pred = model.predict(X)
print(y_pred[:5])
ss_res=np.sum((y-y_pred.flatten())**2)
ss_tot = np.sum((y-np.mean(y))**2)
r2=1-ss_res/ss_tot
print("ssr:R^2 score:", r2)

base_optimizer = SSR(max_iter=20, normalize_columns=True)
model=base_optimizer.fit(X, y)

print("ssr:true coefs ",true_coef)
print("ssr:predicted ", model.coef_[0])







#______________________________________________________________________________________

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython.display import Image
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars

from pysindy.utils import lorenz, lorenz_control, enzyme
import pysindy as ps

# bad code but allows us to ignore warnings
import warnings
from scipy.integrate.odepack import ODEintWarning
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=ODEintWarning)


# Seed the random number generators for reproducibility
np.random.seed(100)

# Initialize integrator keywords for solve_ivp to replicate the odeint defaults
integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12
# define the testing and training Lorenz data we will use for these examples
dt = 0.002

t_train = np.arange(0, 10, dt)
x0_train = [-8, 8, 27]
t_train_span = (t_train[0], t_train[-1])
x_train = solve_ivp(
    lorenz, t_train_span, x0_train, t_eval=t_train, **integrator_keywords
).y.T

t_test = np.arange(0, 15, dt)
t_test_span = (t_test[0], t_test[-1])
x0_test = np.array([8, 7, 15])
x_test = solve_ivp(
    lorenz, t_test_span, x0_test, t_eval=t_test, **integrator_keywords
).y.T

# define the testing and training data for the Lorenz system with control
def u_fun(t):
    return np.column_stack([np.sin(2 * t), t ** 2])


x_train_control = solve_ivp(
    lorenz_control,
    t_train_span,
    x0_train,
    t_eval=t_train,
    args=(u_fun,),
    **integrator_keywords
).y.T
u_train_control = u_fun(t_train)
x_test_control = solve_ivp(
    lorenz_control,
    t_test_span,
    x0_test,
    t_eval=t_test,
    args=(u_fun,),
    **integrator_keywords
).y.T
u_test_control = u_fun(t_test)

feature_names = ['x', 'y', 'z']
ode_lib = ps.WeakPDELibrary(
#     library_functions=library_functions,
#     function_names=library_function_names,
    function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
    spatiotemporal_grid=t_train,
    is_uniform=True,
    K=100,
)

fixmask = np.array([[True, False, False, False, False, True, False, False, False], [False, False, True, False, True, False, False, False, False], [True, True, False, False, True, False, False, False, True]])
fixval = np.array([[-10, 0, 0, 11, 25, 16, 19, 0.5, 110], [11, 20, 23, 0, 11, 25, 1, 0.25, 0.15], [0.5, 120, 110, 1200, 15, 17, 28, 11, 0.06]]) 

sparse_regression_optimizer = ps.STLSQ(threshold=0.1)
  # default is lambda = 0.1
proxy_optimizer = FixedCoefficientOptimizer(
    base_optimizer=sparse_regression_optimizer,
    fixed_coefs=fixmask,
    fixed_values=fixval,
    no_normalization_for_fixeds=False
)
model = ps.SINDy(feature_library=ode_lib, optimizer=proxy_optimizer)
model.fit(x_train, t=dt)
model.print()