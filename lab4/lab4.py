import numpy as np
from sklearn.preprocessing import minmax_scale

x1_min, x1_max = -25, 75
x2_min, x2_max = 25, 65
x3_min, x3_max = 25, 40

y_min = 200 + (x1_min + x2_min + x3_min)/3
y_max = 200 + (x1_max + x2_max + x3_max)/3

n = 8  # кількість точок
c = 8  # кількість коефіцієнтів

norm_factors = np.array([[1, 1, 1],
                         [1, 1, -1],
                         [1, -1, -1],
                         [-1, -1, -1],
                         [-1, 1, 1],
                         [1, -1, 1],
                         [-1, -1, 1],
                         [-1, 1, -1]])

# інтеракції між факторами
def interact_factors(factors):
    full_factors = np.zeros((n, 4))
    full_factors[:, 0] = factors[:, 0] * factors[:, 1]
    full_factors[:, 1] = factors[:, 0] * factors[:, 2]
    full_factors[:, 2] = factors[:, 1] * factors[:, 2]
    full_factors[:, 3] = factors[:, 0] * factors[:, 1] * factors[:, 2]
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

# вирішення нормованого СЛР
def solve_norm_coef(factors, values):
    coefs = np.zeros(c)
    coefs[0] = values.mean()
    for i in range(c-1):
        coefs[i+1] = np.mean(values * factors[:, 0])
    return coefs

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
print(f"Y min: {y_min}\tY max: {y_max}")

# отримання оптимальної кількості випробувань для однієї комбінації факторів
Gt = [6.798, 5.157, 4.737, 3.91]
for i in range(2, 10):
    rand_y = np.random.randint(y_min, y_max, (n, i))
    std_y = np.std(rand_y, axis=1)**2
    print(f"Max Y dispersion: {std_y.max()}")
    Gp = std_y.max()/std_y.mean()
    f1 = i - 1
    print(f"m = {i} Gp: {Gp}\tGt: {Gt[i-2]}")
    if Gp < Gt[i-2]:
        m = i  # кількість випробувань
        break

print(f"Mean Y dispersion: {std_y.mean()}")

# формування факторів
factors = natural(norm_factors)
factors = np.concatenate((factors, interact_factors(factors)), axis=1)
norm_factors = np.concatenate((norm_factors, interact_factors(norm_factors)), axis=1)

# генерація Y значень для експериментів
rand_y = np.random.randint(y_min, y_max, (n, m))
norm_y = minmax_scale(rand_y.reshape(n*m), feature_range=(-1,1)).reshape(n ,m)

# перевірка за критеріїм стюдента
coefs_value = np.zeros(4)
coefs_value[0] = np.mean(rand_y.mean(axis=1))
for i in range(3):
    coefs_value[i+1] = np.mean(rand_y.mean(axis=1) * factors[:, i])
stud_crit = np.abs(coefs_value) / np.sqrt(std_y.mean()/(n*m))
ts = [2.306, 2.12, 2.064]
sig_coefs = len(stud_crit[stud_crit > ts[m-2]])
print(f"All coefs are significant: {sig_coefs == m}\t{sig_coefs}")

# формування планів
norm_plan = np.concatenate((norm_factors, norm_y), axis=1)
norm_plan = np.concatenate((norm_plan, norm_y.mean(axis=1).reshape(n, 1)), axis=1)
std_norm_y = np.std(norm_y, axis=1)**2
norm_plan = np.concatenate((norm_plan, std_norm_y.reshape(n, 1)), axis=1)
print(f"Normalized plan:\n{norm_plan}")
plan = np.concatenate((factors, rand_y), axis=1)
plan = np.concatenate((plan, rand_y.mean(axis=1).reshape(n, 1)), axis=1)
plan = np.concatenate((plan, std_y.reshape(n, 1)), axis=1)
print(f"Natural plan:\n{plan}")

norm_coefs = solve_norm_coef(norm_factors, norm_y.mean(axis=1))
print("Normalized coefs", norm_coefs)
coefs = solve_coef(factors, rand_y.mean(axis=1))
print("Natural coefs", coefs)

reg_val = regression(norm_factors, norm_coefs, n, m)
print("Norm mean Y - regression Y")
print(np.column_stack((norm_y.mean(axis=1), reg_val)))

# перевірка критерію Фішера
f3 = n * (m -1) # 8
f4 = n - sig_coefs # 4
disp = m/(n - sig_coefs) * np.sum((reg_val - norm_y.mean(axis=1))**2)
Fp = disp / std_y.mean()
Ft = 3.8
print(f"Fisher crit: {Fp < Ft}\t{Fp}\t{Ft}")

