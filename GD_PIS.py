import numpy as np
from scipy.optimize import OptimizeResult


def parabolic_interpolation_search(f, a, b, max_iter=1000, tol=1e-6):
    nfev = 0
    nit = 0
    x_min = None

    for nit in range(max_iter):
        midpoint = (a + b) / 2
        f_mid = f(midpoint)
        f_a = f(a)
        f_b = f(b)
        nfev += 3

        numerator = (f_b - f_mid) * (midpoint - a) ** 2 - (f_mid - f_a) * (b - midpoint) ** 2
        denominator = 2 * ((f_b - f_mid) * (midpoint - a) - (f_mid - f_a) * (b - midpoint))

        if np.abs(denominator) < np.finfo(float).eps:
            x_min = midpoint
            break

        x_parabolic = midpoint + numerator / denominator

        if a < x_parabolic < b:
            x_min = x_parabolic
        else:
            x_min = midpoint

        if f(x_min) < f_mid:
            if x_min < midpoint:
                b = midpoint
            else:
                a = midpoint
        else:
            if x_min < midpoint:
                a = x_min
            else:
                b = x_min

        if np.abs(b - a) < tol:
            break

    if x_min is None:
        x_min = (a + b) / 2

    return OptimizeResult(
        x=x_min,
        fun=f(x_min),
        nfev=nfev,
        nit=nit + 1,
    )


def gd_pis(f, grad_f, x0, max_iter=1000, tol=1e-6):
    x = np.array(x0, dtype=float)

    nit = 0
    nfev = 1
    x_history = [x.copy()]

    for nit in range(max_iter):
        g = grad_f(x)

        if np.linalg.norm(g) < tol:
            break

        def f_along_line(alpha):
            return f(x + alpha * -g)

        alpha_result = parabolic_interpolation_search(f_along_line, 0, 1.0, max_iter=max_iter, tol=tol)
        alpha = alpha_result.x
        nfev += alpha_result.nfev

        x_new = x + alpha * -g
        x_history.append(x_new.copy())

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            nfev += 1
            break

        x = x_new
        nfev += 1

    fun = f(x)

    result = OptimizeResult(
        x=x,
        fun=fun,
        nfev=nfev,
        njev=nit + 1,
        nit=nit + 1,
        success=nit < max_iter - 1 or np.linalg.norm(g) < tol,
        x_history=np.array(x_history)
    )

    return result
