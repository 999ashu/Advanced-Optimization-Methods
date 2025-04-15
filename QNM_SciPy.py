from scipy.optimize import minimize, SR1
from helpers import *


def newton_cg_method(f, grad_f, x0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    x_history = [x.copy()]

    def callback(xk):
        x_history.append(xk.copy())

    result = minimize(
        f,
        x0,
        method='Newton-CG',
        jac=grad_f,
        tol=tol,
        options={'maxiter': max_iter},
        callback=callback
    )
    result['x_history'] = np.array(x_history)
    return result


def sr1_method(f, grad_f, x0, tol=1e-6, max_iter=100):
    x_history = [np.array(x0)]

    def callback(xk, res=None):
        x_history.append(xk.copy())

    result = minimize(
        f,
        x0,
        method='trust-constr',
        jac=grad_f,
        tol=tol,
        hess=SR1(),
        options={'maxiter': max_iter},
        callback=callback
    )
    result['x_history'] = np.array(x_history)
    return result


def bfgs_method(f, grad_f, x0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    x_history = [x.copy()]

    def callback(xk):
        x_history.append(xk.copy())

    result = minimize(
        f,
        x0,
        method='BFGS',
        jac=grad_f,
        tol=tol,
        options={'maxiter': max_iter},
        callback=callback
    )
    result['x_history'] = np.array(x_history)
    return result
