import numpy as np

factors = 3
points = 8
max_lim = 20
coefs = [1, 3, 3, 7]  # довільні

# функція відгуку для однієї точки
# повертає значення у точці
def regression(factors, coefs):
    y = coefs[0] + coefs[1] * factors[0] + coefs[2] * factors[1] + coefs[3] * factors[2]
    return y

# застосування функції відгуку до всієї матриці факторів
# повертає масив значень для усіх точок
def func_values(matrix):
    values = np.zeros((points, 1), matrix.dtype)
    for p in range(points):
        values[p] = regression(matrix[p], coefs)
    return values

# нормалізація факторів
# повертає матрицю нормалізованих факторів
def norm_factors(matrix):
    xMin, xMax = matrix.min(0), matrix.max(0)
    x0 = (xMax + xMin)/2
    dx = x0 - xMin

    norm = np.zeros((points, factors), float)
    for p in range(points):
        norm[p] = (matrix[p] - x0)/dx
    return norm

# отримання індексу точки, що відповідає критерію
# повертає елемент що має найменшу різницю з середнім значенням по модулю
def point_by_crit(values):
    return np.abs(values-values.mean()).argmin()

# генерація матриці факторів з випадкових чисел
matrix = np.random.randint(0, max_lim, factors*points, int).reshape((points, factors))

# запис матриці з нормалізованих факторів та їх значень за фунцією відгуку
norm = norm_factors(matrix)
norm = np.append(norm, func_values(norm), axis=1)

# додаванн до матриці факторів колонки з їх значеннями
matrix = np.append(matrix, func_values(matrix), axis=1)
# індекс точки та її фактори, що задовільняють критерій вибору
best_point = point_by_crit(matrix[:, -1])
best_factors = matrix[best_point, :-1]

print("Matrix:")
print(matrix)

print("Normalized:")
print(norm)

print("Point fits crit -> Y; Y = mean(y)")
print(f"Index: {best_point}")
print(f"Factors: X1 = {best_factors[0]}, X2 = {best_factors[1]}, X3 = {best_factors[2]}")
print(f"Y = {coefs[0]} + {coefs[1]}*{best_factors[0]} + {coefs[2]}*{best_factors[1]} + {coefs[3]}*{best_factors[2]}")
