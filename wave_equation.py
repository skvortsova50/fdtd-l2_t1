import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ============================================================
#                  СПЕКТРАЛЬНІ ФУНКЦІЇ
# ============================================================

def spectrum(signal, dt):
    signal = signal - np.mean(signal)
    SPEC = np.abs(np.fft.fftshift(np.fft.fft(signal)))
    freq = np.fft.fftshift(np.fft.fftfreq(len(signal), d=dt))
    return SPEC, freq


def analytic_spectrum(freq, pulse_type, sigma_t):
    if pulse_type == "gaussian":
        return np.exp(-2 * np.pi**2 * sigma_t**2 * freq**2)

    elif pulse_type == "rectangular":
        return np.abs(np.sinc(freq * sigma_t))

    elif pulse_type == "triangular":
        return np.abs(np.sinc(freq * sigma_t))**2


# ============================================================
#                     ІМПУЛЬСИ
# ============================================================

def gaussian_pulse(x, mean, sigma):
    return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


def rectangular_pulse(x, mean, sigma):
    return np.where(np.abs(x - mean) <= sigma / 2, 1.0, 0.0)


def triangular_pulse(x, mean, sigma):
    return np.where(np.abs(x - mean) <= sigma / 2,
                    1.0 - 2.0 * np.abs(x - mean) / sigma,
                    0.0)


# ============================================================
#                 КЛАС FDTD-СИМУЛЯЦІЇ
# ============================================================

class FDTD1D:
    """
    1D FDTD для хвильового рівняння:
        u_tt = c^2 u_xx
    """

    def __init__(self, x_dim, time_tot, c, dx, S,
                 obs_probe, record_stop_time):

        self.x_dim = x_dim
        self.time_tot = time_tot
        self.c = c
        self.dx = dx
        self.S = S
        self.dt = S * dx / c

        self.u_past = np.zeros(x_dim)
        self.u_present = np.zeros(x_dim)
        self.u_future = np.zeros(x_dim)

        self.obs_probe = obs_probe
        self.record_stop_time = record_stop_time
        self.signal_obs = np.zeros(time_tot)

        self.t_step = 0

    def set_initial(self, u0, u1):
        self.u_past[:] = u0
        self.u_present[:] = u1

    def step(self):
        u = self.u_present
        up = self.u_past
        uf = self.u_future

       uf[2:-2] = (2 * u[2:-2] - up[2:-2] + 
                   (self.S**2 / 12) * (-u[4:] + 16 * u[3:-1] - 30 * u[2:-2] + 16 * u[1:-3] - u[:-4]))


uf[0] = 0
uf[1] = 0
uf[-1] = 0
uf[-2] = 0

        if self.t_step < self.record_stop_time:
            self.signal_obs[self.t_step] = u[self.obs_probe]

        self.u_past[:] = self.u_present
        self.u_present[:] = self.u_future
        self.t_step += 1

    def run(self):
        for _ in range(self.time_tot):
            self.step()


# ============================================================
#                ОСНОВНИЙ СЦЕНАРІЙ
# ============================================================

# --- Параметри моделювання ---
x_dim = 200
time_tot = 400
c = 1.0
dx = 1.0
S = 1

source_pos = x_dim // 2
obs_probe = int(0.75 * x_dim)
record_stop_time = 100

x = np.arange(0, x_dim, dx)

# --- Імпульс ---
sigma_0 = 5
pulse = rectangular_pulse
pulse_name = "rectangular_pulse"

u0 = pulse(x, source_pos, sigma_0)
u1 = pulse(x - dx, source_pos, sigma_0)

# --- Ініціалізація симуляції ---
sim = FDTD1D(x_dim, time_tot, c, dx, S,
             obs_probe, record_stop_time)
sim.set_initial(u0, u1)

# ============================================================
#                      АНІМАЦІЯ
# ============================================================

fig, ax = plt.subplots()
ax.set_xlim(0, x_dim)
ax.set_ylim(-1.2, 1.2)
ax.set_title("1D FDTD: хвильове рівняння")
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
line, = ax.plot([], [], lw=2)


def update(frame):
    sim.step()
    line.set_data(np.arange(sim.x_dim), sim.u_present)
    return line,


ani = animation.FuncAnimation(fig, update,
                              frames=time_tot,
                              interval=20,
                              blit=True,
                              repeat=False)

plt.show()

# ============================================================
#                 СПЕКТРАЛЬНИЙ АНАЛІЗ
# ============================================================

t = np.arange(0, time_tot * sim.dt, sim.dt)

SPEC_obs, freq = spectrum(sim.signal_obs, sim.dt)

sigma_t = sigma_0 * dx / c
SPEC_analytic = analytic_spectrum(freq, pulse_name, sigma_t)

pos = freq > 0

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t, sim.signal_obs)
plt.axvline(record_stop_time * sim.dt, color='r', linestyle='--')
plt.title("Сигнал у точці спостереження")
plt.xlabel("t")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(freq[pos],
         SPEC_obs[pos] / SPEC_obs[pos].max(),
         'o', markersize=4, label="FDTD (FFT)")
# plt.plot(freq[pos], SPEC_analytic[pos],
#          '-', linewidth=2, label="Аналітичний")

N_T_vals = [5, 10, 20]
for N_T in N_T_vals:
    plt.axvline(1.0 / (N_T * sim.dt),
                linestyle='--', label=f"N_T = {N_T}")

plt.title("Спектри")
plt.xlabel("f")
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
