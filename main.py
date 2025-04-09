import math
import numpy as np
import pandas as pd

# ======= MAX VALUES =========
X_MIN, X_MAX = 16, 40
W_MIN, W_MAX = 0, 100

# ======== MODEL FUNCTIONS =========
def membership(x_j, a_ij, b_ij):
    # return math.exp(- ((x_j - a_ij) / b_ij) ** 2)
    return max(0, 1 - ((2 * abs(x_j - a_ij)) / b_ij))

def mu(memberships):
    return np.prod(memberships)

def E(y, yr):
    return 1/2 * (y - yr)**2

def sgn(x):
    return 1 if x > 0 else -1

def a_next(aij, ka, mu, y, yr, wi, bij, xj, mem):
    if mem == 0:
        return aij  # Handle zero div
    return aij - ka * (y - yr) * (wi - y) * sgn(xj - aij) * (2 / (bij * mem))

def b_next(bij, kb, mu, y, yr, wi, mem):
    if mem == 0:
        return bij  # Handle zero div
    return bij - kb * (y - yr) * (wi - y) * ((1 - mem) / mem) * (1 / bij)

def wi_next(wi, kw, mu, membership_values, y, yr):
    mus = sum(membership_values)
    if mus == 0:
        return wi  # Handle zero div
    return wi - ((kw * mu) / mus) * (y - yr)

def yf(membership_values, wi):
    mus = sum(membership_values)
    if mus == 0:
        return 0
    res = sum(membership_values[i] * wi[i] for i in range(len(wi)))
    return res / mus

# ======== ENTRENAMIENTO =========
def entrenar(reglas, wi, data, ka=0.1, kb=0.1, kw=0.1, epocas=1000):
    for epoch in range(epocas):
        for x_1, yr in data:
            membership_values = [membership(x_1, r[0], r[1]) for r in reglas]
            y_inferida = yf(membership_values, wi)

            for i in range(len(reglas)):
                reglas[i][0] = a_next(reglas[i][0], ka, membership_values[i], y_inferida, yr, wi[i], reglas[i][1], x_1, membership_values[i])
                reglas[i][1] = b_next(reglas[i][1], kb, membership_values[i], y_inferida, yr, wi[i], membership_values[i])

            for i in range(len(wi)):
                wi[i] = wi_next(wi[i], kw, membership_values[i], membership_values, y_inferida, yr)

    return reglas, wi

# ======== INFERENCE =========
def inferir(x_1, reglas, wi):
    membership_values = [membership(x_1, r[0], r[1]) for r in reglas]

    # Debug print
    # print(f"Pertenencia -> Baja: {membership_values[0]}%, Media: {membership_values[1]}%, Alta: {membership_values[2]}%")

    return yf(membership_values, wi)

# ======== SCALE ========
def scale_x(x):
    return (x - X_MIN) / (X_MAX - X_MIN)

def scale_w(w):
    return (w - W_MIN) / (W_MAX - W_MIN)

def unscale_w(w):
    return w * (W_MAX -W_MIN) + W_MIN

# ======== MAIN =========
if __name__ == '__main__':
    # Training Data (temp, % vent)
    # data = [
    #     (18, 20),
    #     (19, 22),
    #     (20, 24),
    #     (21, 26),
    #     (22, 28),
    #     (23, 30),
    #     (24, 32),
    #
    #     (25, 40),
    #     (26, 45),
    #     (27, 48),
    #     (28, 50),
    #     (29, 53),
    #     (30, 55),
    #     (31, 58),
    #
    #     (32, 65),
    #     (33, 68),
    #     (34, 70),
    #     (36, 74),
    #     (38, 78),
    #     (40, 80)
    # ]

    data = pd.read_csv("test/datos_entrenamiento_16_40.csv").values.tolist()

    # Normalize data
    data = [ (scale_x(x), scale_w(w)) for x, w in data]

    # Initialize Rules
    reglas = [
        [20, 15, 25],
        [28, 15, 50],
        [36, 15, 75]
    ]
    wi = [r[2] for r in reglas]

    # Training
    reglas, wi = entrenar(reglas, wi, data, epocas=50000)

    # Mostrar parámetros entrenados
    print("Parámetros después del entrenamiento:")
    for i, r in enumerate(reglas):
        print(f"Regla {i+1}: a={r[0]:.3f}, b={r[1]:.3f}, w={wi[i]:.3f}")

    # Inferencia con entrada por consola
    while True:
        entrada = input("\nIngresa una temperatura (o 'salir' para terminar): ")
        if entrada.lower() == 'salir':
            break
        try:
            x = float(entrada)
            salida = inferir(x, reglas, wi)
            # salida = unscale_w(salida)
            print(f"Salida inferida: {salida:.2f}% ventilador")
        except ValueError:
            print("Por favor ingresa un número válido.")
