import numpy as np
import random

x1_min, x1_max = 10, 50
x2_min, x2_max = -20, 60
x3_min, x3_max = -20, 20

x01 = (x1_min + x1_max)/2
x02 = (x2_min + x2_max)/2
x03 = (x3_min + x3_max)/2

l = 1.73  # значення для зоряних точок плану

n = 14  # кількість точок
c = 11  # кількість коефіцієнтів
m = 2  # кількість випробувань

norm_factors = np.array([[1, 1, 1],
                         [1, 1, -1],
                         [1, -1, -1],
                         [-1, -1, -1],
                         [-1, 1, 1],
                         [1, -1, 1],
                         [-1, -1, 1],
                         [-1, 1, -1],
                         [-l, 0, 0],
                         [l, 0, 0],
                         [0, -l, 0],
                         [0, l, 0],
                         [0, 0, -l],
                         [0, 0, l]])

# генерація значень Y
def generate_y(factors, n, m):
    Y = np.zeros((n, m))
    for r in range(n):
        x1, x2, x3 = factors[r, 0:3]
        for i in range(m):
            Y[r, i] = 8.8 + 8.0 * x1 + 5.4 * x2 + 8.0 * x3 + 0.2 * x1 * x1 + \
                      0.2 * x2 * x2 + 2.9 * x3 * x3 + 3.4 * x1 * x2 + 0.9 * \
                      x1 * x3 + 3.5 * x2 * x3 + 0.3 * x1 * x2 * x3 + random.randint(0,10) - 5
    return Y


# інтеракції між факторами
def interact_factors(factors, expanded=False):
    full_factors = np.zeros((n, 4)) if not expanded else np.zeros((n, 7))
    full_factors[:, 0] = factors[:, 0] * factors[:, 1]
    full_factors[:, 1] = factors[:, 0] * factors[:, 2]
    full_factors[:, 2] = factors[:, 1] * factors[:, 2]
    full_factors[:, 3] = factors[:, 0] * factors[:, 1] * factors[:, 2]
    if expanded:
        for i in range(3):
            full_factors[:, 4+i] = np.power(factors[:, i], 2)
    return full_factors


# отримання натуральних факторів з нормованих
def natural(norm_factors):
    natural = np.zeros_like(norm_factors)
    natural[:, 0][norm_factors[:, 0] == 1] = x1_max
    natural[:, 1][norm_factors[:, 1] == 1] = x2_max
    natural[:, 2][norm_factors[:, 2] == 1] = x3_max

    natural[:, 0][norm_factors[:, 0] == -1] = x1_min
    natural[:, 1][norm_factors[:, 1] == -1] = x2_min
    natural[:, 2][norm_factors[:, 2] == -1] = x3_min

    natural[:, 0][norm_factors[:, 0] == 0] = x01
    natural[:, 1][norm_factors[:, 1] == 0] = x02
    natural[:, 2][norm_factors[:, 2] == 0] = x03

    natural[:, 0][norm_factors[:, 0] == l] = l * (x1_max - x01) + x01
    natural[:, 1][norm_factors[:, 1] == l] = l * (x2_max - x02) + x02
    natural[:, 2][norm_factors[:, 2] == l] = l * (x3_max - x03) + x03
    natural[:, 0][norm_factors[:, 0] == -l] = -l * (x1_max - x01) + x01
    natural[:, 1][norm_factors[:, 1] == -l] = -l * (x2_max - x02) + x02
    natural[:, 2][norm_factors[:, 2] == -l] = -l * (x3_max - x03) + x03
    return natural


# регресія за факторами та коефцієнтами
def regression(matrix, coefs, n, c):
    def func(factors, coefs, c):
        y = coefs[0]
        for i in range(c):
            y += coefs[i+1] * factors[i]
        return y
    values = np.zeros(n)
    for f in range(n):
        values[f] = func(matrix[f], coefs, c)
    return values


# вирішення натурального СЛР
def solve_coef(factors, values):
    A = np.zeros((c, c))
    x_mean = np.mean(factors, axis=0)
    for i in range(1, c):
        for j in range(1, c):
            A[j, i] = np.mean(factors[:, i-1] * factors[:, j-1])
    A[0, 0] = n
    A[0, 1:] = x_mean
    A[1:, 0] = x_mean
    B = np.mean(np.tile(values, (c, 1)).T * np.column_stack((np.ones(n), factors)), axis=0)
    coef = np.linalg.solve(A, B)
    return coef


print(f"X1 min: {x1_min}\tX1 max: {x1_max}")
print(f"X2 min: {x2_min}\tX2 max: {x2_max}")
print(f"X3 min: {x3_min}\tX3 max: {x3_max}")

factors = natural(norm_factors)

# Перевірка однорідності дисперсії за критерієм Кохрена
# отримання оптимальної кількості випробувань для однієї комбінації факторів
Gt = 0.3346
for i in range(m, 10):
    rand_y = generate_y(factors, n, i)
    std_y = np.std(rand_y, axis=1)**2
    print(f"Max Y dispersion: {std_y.max()}")
    Gp = std_y.max()/std_y.mean()
    print(f"m = {i} Gp: {Gp}\tGt: {Gt}")
    if Gp < Gt:
        m = i  # кількість випробувань
        break

print(f"Mean Y dispersion: {std_y.mean()}")
factors = np.concatenate((factors, interact_factors(factors, True)), axis=1)

# перевірка за критеріїм стюдента
coefs_value = np.zeros(4)
coefs_value[0] = np.mean(rand_y.mean(axis=1))
for i in range(3):
    coefs_value[i+1] = np.mean(rand_y.mean(axis=1) * factors[:, i])
stud_crit = np.abs(coefs_value) / np.sqrt(std_y.mean()/(n*m))
ts = 2.048
sig_ind = np.argwhere(stud_crit > ts)

print(f"All coefs are significant: {len(sig_ind.flatten()) == c}\t{sig_ind.flatten()}")

# формування планів
plan = np.concatenate((factors, rand_y), axis=1)
plan = np.concatenate((plan, rand_y.mean(axis=1).reshape(n, 1)), axis=1)
plan = np.concatenate((plan, std_y.reshape(n, 1)), axis=1)
print(f"Natural plan:\n{plan}")

coefs = solve_coef(factors, rand_y.mean(axis=1))
print("Natural coefs", coefs)
significant_coefs = np.zeros_like(coefs)
significant_coefs[sig_ind] = coefs[sig_ind]
print("Significant coefs", significant_coefs)

reg_val = regression(factors, significant_coefs, n, len(sig_ind.flatten()))
print("Natural mean Y - regression Y")
print(np.column_stack((rand_y.mean(axis=1), reg_val)))

# перевірка критерію Фішера
disp = m/(n - len(sig_ind.flatten())) * np.sum((reg_val - rand_y.mean(axis=1))**2)
Fp = disp / std_y.mean()
Ft = 3.0
print(f"Fisher crit: {Fp < Ft}\t{Fp}\t{Ft}")

