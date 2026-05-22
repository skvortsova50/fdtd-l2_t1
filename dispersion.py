import numpy as np
import matplotlib.pyplot as plt


# Діапазон N_lambda
N_lambda = np.linspace(2, 20, 100)  # починати не з нуля!

# Набір значень S
S_values = [0.7, 0.9, 0.95, 0.99, 1.0, 1.1]

def v_of_N(Nl, S):
    return np.pi/ (Nl * np.arcsin(np.sin(np.pi * S / Nl) / S))

plt.figure(figsize=(8, 5))

for S in S_values:
    v = v_of_N(N_lambda, S)
    plt.plot(N_lambda, v, label=f"S = {S}")

plt.xlabel(r"$N_\lambda$")
plt.ylabel(r"$v(N_\lambda)$")
plt.title(r"Залежність $v(N_\lambda)/c$ для різних $S$")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
