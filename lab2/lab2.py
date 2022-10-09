import math
import numpy as np

n = 3
k = 2
m = 5

x1_min, x1_max = 10, 50
x2_min, x2_max = -20, 60
y_min, y_max = 50, 150

X_norm = np.array([[-1, -1],
                   [-1, 1],
                   [1, -1]])

X = np.array([[x1_min, x1_max, (x1_min + x1_max)/2, abs(x1_max - x1_min)/2],
              [x2_min, x2_max, (x2_min + x2_max)/2, abs(x2_max - x2_min)/2]])

Rcrit = 2.16


# функція регресії для матриці факторів
def regression(matrix, coefs, n, k):
    def func(factors, coefs, k):
        y = coefs[0]
        for i in range(k):
            y += coefs[i+1] * factors[i]
        return y
    values = np.zeros(n)
    for f in range(n):
        values[f] = func(matrix[f], coefs, k)
    return values


# генерація рандомних значень до відпоідності критерію Романовського
def generate_val(n, m):
    roman = False
    while not roman:
        val = np.random.uniform(y_min, y_max, (n, m))  # значення
        dispers = val.std(axis=1)
        F = np.zeros(n)
        for i in range(n):
            j = i + 1 if i != 2 else 0
            s1, s2 = dispers[i], dispers[j]
            F[i] = s1/s2 if s1 >= s2 else s2/s1
        R = np.abs((m-2)/m * F - 1)/math.sqrt((2*(2*m-2))/(m*(m-4)))  # експериментальне значення критерію Романовського
        roman = True
        for r in R:
            if r > Rcrit:  # перевірка на критерій Романовського
                roman = False
                m += 1
                print(f"{m} experiments distribution doesn't fit in Roman crit")
    return val, m


# вірішення СЛР для коефіцієнтва по факторам та значенням
def solve_coef(factors, values):
    x_mean = factors.mean(axis=0)
    y_mean = values.mean()
    a = np.power(factors, 2).mean(axis=0)
    a2 = np.mean(factors[:, 0] * factors[:, 1])
    A = np.array([[1, x_mean[0], x_mean[1]],
                   [x_mean[0], a[0], a2],
                   [x_mean[1], a2, a[1]]])
    B = np.array([y_mean, np.mean(factors[:, 0] * y_mean), np.mean(factors[:, 1] * y_mean)])
    coef = np.linalg.solve(A, B)
    # print(coef)
    # b = np.zeros(n)
    # for i in range(n):
    #     linal = A.copy()
    #     linal[:, i] = B
    #     b[i] = np.linalg.det(linal)/np.linalg.det(A)
    # print(b)
    return coef


print(f"Plan factors min, max, mean, dispers:\n{X}")
print(f"Normalized plan factors:\n{X_norm}")
val, m = generate_val(n, m)
print(f"Generated Y values:\n{val}")
print(f"Mean Y:\n{val.mean(axis=1)}")
b = solve_coef(X_norm, val)
print(f"Solved SLE for b:\n{b}")
a = [b[0] + b[1]*X[0, 2]/X[0, 3] + b[1]*X[1, 2]/X[1, 3], b[1]/X[0, 3], b[2]/X[1, 3]]
print(f"Naturalized b as a:\n{a}")
print(f"Regression for b and normalized X:\n{regression(X_norm, b, n, k)}")
print(f"Regression for a and X:\n{regression([[10, -20], [10, 60], [50, -20]], a, n, k)}")

