from helpers import *
from scipy.optimize import OptimizeResult


def gd_lrs(f, grad_f, x0, learning_rate_schedule, tol=1e-6, max_iter=1000):
    x = np.array(x0, dtype=float)

    nit = 0
    x_history = [x.copy()]

    for nit in range(max_iter):
        lr = learning_rate_schedule(nit)
        grad = grad_f(x)
        x -= lr * grad

        x_history.append(x.copy())

        if np.linalg.norm(grad) < tol:
            break

    return OptimizeResult(x=x,
                          fun=f(x),
                          nfev=0,
                          njev=nit + 1,
                          nit=nit + 1,
                          x_history=np.array(x_history))


def const_decay(decay_rate):
    return decay_rate


def step_decay(init, drop_rate, step_size, nit):
    return init * (drop_rate ** (nit // step_size))


def exponential_decay(init, decay_rate, nit):
    return init * np.exp(-decay_rate * nit)


def cosine_annealing(init, nit, lr_max, eta_min=0):
    return eta_min + (init - eta_min) * (1 + np.cos(np.pi * nit / lr_max)) / 2


decay_modes = [(lambda k: const_decay(0.1), "GD const decay"),
               (lambda k: step_decay(0.1, 0.5, 10, k), "GD step decay"),
               (lambda k: exponential_decay(0.1, 0.01, k), "GD exponential decay"),
               (lambda k: cosine_annealing(0.1, k, 100, eta_min=0.01), "GD cosine annealing")]

# for decay in decay_modes:
#     print_output([-3, -25],
#                  lambda *args, **kwargs: gd_lrs(*args, learning_rate_schedule=decay[0], **kwargs),
#                  func_re,
#                  grad_re,
#                  [-18, 16],
#                  decay[1],
#                  grid=[-30, 30])
