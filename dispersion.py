import numpy as np
import matplotlib.pyplot as plt


# Діапазон N_lambda
N_lambda = np.linspace(2, 20, 100)  # починати не з нуля!

# Набір значень S
S_values = [0.1, 0.3, 0.5, 0.7, 0.85]

def v_of_N_4th(Nl, S):
    term = ((4/3) * np.sin(np.pi / Nl)**2 - (1/12) * np.sin(2 * np.pi / Nl)**2)

    omega_dt = 2 * np.arcsin(S * np.sqrt(term))

    return (omega_dt * Nl) / (2 * np.pi * S)

plt.figure(figsize=(8, 5))

for S in S_values:
    v = v_of_N_4th(N_lambda, S)
    plt.plot(N_lambda, v, label=f"S = {S}")

plt.xlabel(r"$N_\lambda$")
plt.ylabel(r"$v(N_\lambda)/c$")
plt.title(r"Дисперсія для схеми 4-го порядку")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
