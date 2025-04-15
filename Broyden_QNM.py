from scipy.optimize import OptimizeResult
from helpers import *


def broyden_method_implementation(f, grad_f, x0, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)
    n = len(x)
    B = np.eye(n)
    nfev = 1
    njev = 0
    x_history = [x.copy()]
    g = grad_f(x)
    njev += 1

    for nit in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
        try:
            p = -np.linalg.solve(B, g)
        except np.linalg.LinAlgError:
            p = -g

        if np.dot(p, g) > 0:
            p = -g
        alpha = 1.0
        x_new = x + alpha * p
        f_new = f(x_new)
        f_current = f(x)
        nfev += 2
        backtrack_ratio = 0.5
        backtrack_max_iter = 20
        backtrack_iter = 0

        while f_new > f_current and backtrack_iter < backtrack_max_iter:
            alpha *= backtrack_ratio
            x_new = x + alpha * p
            f_new = f(x_new)
            nfev += 1
            backtrack_iter += 1

        s = alpha * p
        x_history.append(x_new.copy())

        g_new = grad_f(x_new)
        njev += 1
        y = g_new - g

        # Формула Бройдена для обновления гессиана: B_{k+1} = B_k + ((y - B_k*s) * s^T) / (s^T * s)
        if np.dot(s, s) > np.finfo(float).eps:
            B = B + np.outer(y - B.dot(s), s) / np.dot(s, s)

        if np.linalg.norm(x_new - x) < tol:
            x = x_new
            break
        x = x_new
        g = g_new
    fun = f(x)

    result = OptimizeResult(
        x=x,
        fun=fun,
        nfev=nfev,
        njev=njev,
        nit=nit + 1,
        x_history=np.array(x_history)
    )
    return result
