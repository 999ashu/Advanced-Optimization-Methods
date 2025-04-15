import numpy as np
import math
import matplotlib.pyplot as plt


## Plot builder
def plot(func, results_list, grid, label):
    if not isinstance(results_list, list):
        results_list = [results_list]

    x, y = np.meshgrid(np.linspace(grid[0], grid[1], 200), np.linspace(grid[0], grid[1], 200))

    z = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i, j] = func((x[i, j], y[i, j]))

    plt.figure()
    plt.contour(x, y, z, levels=30)

    for res in results_list:
        path = res.x_history
        plt.plot(path[:, 0], path[:, 1], label=label, marker='o')
        plt.plot(path[-1, 0], path[-1, 1], 'x', markersize=10)

    plt.xlim(grid)
    plt.ylim(grid)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


## General output formatter
def print_output(init, gd, func, grad, optimum, label=None, grid=(-6, 6)):
    res = gd(func, grad, init)
    error = np.linalg.norm(res.x - optimum)

    print(f"{'=' * 90}")
    print(f"Method: {label is not None and label or gd.__name__}")
    print(f"Initial point: {init}")
    print(f"Real minimum: {optimum}")
    print(f"{'=' * 90}")
    print(f"{'Method':<15} {'Found point':<20} {'f(x)':<10} {'Iterations':<15} {'Evals':<12} {'Error':<10}")
    print(f"{'-' * 90}")
    print(
        f"{'Stat':<15} ({res.x[0]:.3f}, {res.x[1]:.3f}) {' ':<5} {res.fun:<13.3f} {res.nit:<13} {res.nfev + res.njev:<10} {error:<10.6f}")
    print(f"{'=' * 90}")

    plot(func, res, grid, label is not None and label or gd.__name__)


def print_output_multi_minima(init, gd, func, grad, optima_list, label=None, grid=(-6, 6)):
    res = gd(func, grad, init)

    min_error = float('inf')
    closest_optimum = None
    for optimum in optima_list:
        error = np.linalg.norm(res.x - optimum)
        if error < min_error:
            min_error = error
            closest_optimum = optimum

    print(f"{'=' * 90}")
    print(f"Method: {label is not None and label or gd.__name__}")
    print(f"Initial point: {init}")
    print(f"Known minima: {optima_list}")
    print(f"Closest minimum: {closest_optimum}")
    print(f"{'=' * 90}")
    print(f"{'Method':<15} {'Found point':<20} {'f(x)':<10} {'Iterations':<15} {'Evals':<12} {'Error':<10}")
    print(f"{'-' * 90}")
    print(
        f"{'Stat':<15} ({res.x[0]:.3f}, {res.x[1]:.3f}) {' ':<5} {res.fun:<13.3f} {res.nit:<13} {res.nfev + res.njev:<10} {min_error:<10.6f}")
    print(f"{'=' * 90}")

    plot(func, res, grid, label is not None and label or gd.__name__)


## Rotated elliptical function
def func_re(point):
    x, y = point
    return (2 * (x + 2) ** 2) + (4 * x * y) + (3 * (y - 4) ** 2)


def grad_re(point):
    x, y = point
    return np.array([4 * (x + 2) + 4 * y, 4 * x + 6 * (y - 4)])


## Rosenbrock function
def rosenbrock(point):
    x, y = point
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def grad_rosenbrock(point):
    x, y = point
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2),
                     200 * (y - x ** 2)])


## Himmelblau function
himmelblau_minima = [
    [3, 2],
    [-2.805118, 3.131312],
    [-3.779310, -3.283186],
    [3.584428, -1.848126]
]

def himmelblau(point):
    x, y = point
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2


def grad_himmelblau(point):
    x, y = point
    return np.array([4 * (x ** 2 + y - 11) * x + 2 * (x + y ** 2 - 7),
                     4 * (x ** 2 + y - 11) * y + 2 * (x + y ** 2 - 7)])


# Bukin function N 6
def bukin(point):
    x, y = point
    return 100 * np.sqrt(np.abs(y - 0.01 * x ** 2)) + 0.01 * np.abs(x + 10)


def grad_bukin(point):
    x, y = point
    return np.array([0.01 * (x + 10) / np.sqrt(np.abs(y - 0.01 * x ** 2)) + 100 * np.sign(y - 0.01 * x ** 2),
                     100 / (2 * np.sqrt(np.abs(y - 0.01 * x ** 2)))])


# Rastrigin multimodal function
def rastrigin(point, a=1):
    x, y = point
    return (a * 2
            + (x ** 2 - a * math.cos(2 * math.pi * x))
            + (y ** 2 - a * math.cos(2 * math.pi * y)))


def grad_rastrigin(point, a=1):
    x, y = point
    return np.array([2 * x + a * 2 * np.pi * np.sin(2 * np.pi * x),
                     2 * y + a * 2 * np.pi * np.sin(2 * np.pi * y)])
