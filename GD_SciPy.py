from helpers import *
import numpy as np
import scipy.optimize as opt


def gd_dlr_opt(func, grad, init, max_iter=1000, tol=1e-6):
    x_history = [np.array(init, dtype=float)]

    def callback(xk):
        x_history.append(np.array(xk, dtype=float))

    result = opt.minimize(func, x0=init, jac=grad, method='BFGS', tol=tol, callback=callback,
                          options={'maxiter': max_iter})

    result.x_history = np.array(x_history)

    return result


def gd_line_search_opt(func, grad, init, max_iter=1000, tol=1e-6):
    x = np.array(init, dtype=float)

    x_history = [x.copy()]
    nit = 0
    nfev = 0

    for i in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break

        def line_search_func(alpha):
            nonlocal nfev
            nfev += 1
            return func(x - alpha * g)

        res = opt.minimize_scalar(line_search_func, method='golden', tol=tol)

        x = x - res.x * g

        x_history.append(x.copy())
        nit += 1
        nfev += res.nfev + 1

        if np.linalg.norm(x_history[-1] - x_history[-2]) < tol:
            break

    return opt.OptimizeResult(
        x=x,
        fun=func(x),
        nit=nit + 1,
        nfev=nfev,
        njev=nit + 1,
        x_history=np.array(x_history),
    )
