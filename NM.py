from GD_GSS import golden_section_search
from GD_LRS import *
from helpers import *


def newton_method(f, grad_f, x0, tol=1e-6, max_iter=1000, line_search_method='golden', lr_params=None):
    x = np.array(x0, dtype=float)
    eps = np.sqrt(np.finfo(float).eps)

    nfev = 1
    njev = 0
    x_history = [x.copy()]

    for nit in range(max_iter):
        g = grad_f(x)
        njev += 1

        if np.linalg.norm(g) < tol:
            break

        n = len(x)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_i_plus = x.copy()
                x_i_plus[i] += eps
                x_j_plus = x.copy()
                x_j_plus[j] += eps
                x_ij_plus = x.copy()
                x_ij_plus[i] += eps
                x_ij_plus[j] += eps

                H[i, j] = (f(x_ij_plus) - f(x_i_plus) - f(x_j_plus) + f(x)) / (eps * eps)
                nfev += 4

        H = (H + H.T) / 2

        try:
            p = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            p = -g

        if np.dot(p, g) > 0:
            p = -g

        if line_search_method == 'golden':
            def f_along_line(alpha):
                return f(x + alpha * p)

            alpha_result = golden_section_search(f_along_line, 0, 1.0)
            alpha = alpha_result.x
            nfev += alpha_result.nfev
        else:
            alpha = lr_params['lr_func'](nit)

        x_new = x + alpha * p
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
        njev=njev,
        nit=nit + 1,
        x_history=np.array(x_history)
    )

    return result


decay_modes = [(lambda k: const_decay(0.1), "GD const decay"),
               (lambda k: step_decay(0.1, 0.5, 10, k), "GD step decay"),
               (lambda k: exponential_decay(0.1, 0.01, k), "GD exponential decay"),
               (lambda k: cosine_annealing(0.1, k, 100, eta_min=0.01), "GD cosine annealing")]
