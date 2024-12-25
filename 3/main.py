import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.lines as lines

m1 = 1.0
m2 = 0.6
l1 = 1.0
l2 = 1.0
a = 0.4
c = 100.0
phi0 = np.pi/4
g = 9.81

def spring_force(phi1, phi2):
    xD = a * np.cos(phi1)
    yD = a * np.sin(phi1)
    xE = a * np.cos(phi2)
    yE = a * np.sin(phi2)
    L = np.sqrt((xE - xD)**2 + (yE - yD)**2)
    L0 = 2 * a * np.sin(phi0/2)
    return c * (L - L0)

def system_eqs(state, t):
    phi1, phi2, dphi1_dt, dphi2_dt = state
    I1 = m1 * l1**2 / 3
    I2 = m2 * l2**2 / 3
    Fs = spring_force(phi1, phi2)
    spring_term = a * c * (np.sin(phi2 - phi1) - 2 * np.sin(phi0/2) * np.cos(phi2/2) * np.sin(phi1/2))
    gravity_term1 = (m1 * g * l1/2) * np.sin(phi1)
    gravity_term2 = (m2 * g * l2/2) * np.sin(phi2)
    d2phi1_dt2 = (spring_term - gravity_term1) / I1
    d2phi2_dt2 = (-spring_term - gravity_term2) / I2
    return [dphi1_dt, dphi2_dt, d2phi1_dt2, d2phi2_dt2]

t = np.linspace(0, 10, 1000)
initial_state = [phi0 - 0.1, phi0 + 0.1, 0, 0]
solution = odeint(system_eqs, initial_state, t)
phi1_sol = solution[:, 0]
phi2_sol = solution[:, 1]
dphi1_dt_sol = solution[:, 2]
dphi2_dt_sol = solution[:, 3]

def calculate_reactions(t, solution):
    phi1 = solution[:, 0]
    phi2 = solution[:, 1]
    dphi1_dt = solution[:, 2]
    dphi2_dt = solution[:, 3]
    d2phi1_dt2 = np.gradient(dphi1_dt, t)
    d2phi2_dt2 = np.gradient(dphi2_dt, t)
    Rx = -0.5 * (m1*l1*(dphi1_dt**2*np.cos(phi1) + d2phi1_dt2*np.sin(phi1)) +
                 m2*l2*(dphi2_dt**2*np.cos(phi2) + d2phi2_dt2*np.sin(phi2)))
    Ry = 0.5 * (m1*l1*(dphi1_dt**2*np.sin(phi1) - d2phi1_dt2*np.cos(phi1)) +
                m2*l2*(dphi2_dt**2*np.sin(phi2) - d2phi2_dt2*np.cos(phi2))) + (m1 + m2)*g
    return Rx, Ry

Rx, Ry = calculate_reactions(t, solution)

fig_anim = plt.figure(figsize=(8, 8))
ax_anim = fig_anim.add_subplot(111)
ax_anim.set_xlim(-2, 2)
ax_anim.set_ylim(-2, 2)
ax_anim.grid(True)
ax_anim.set_aspect('equal')
ax_anim.set_title('Анимация системы')
line_OA, = ax_anim.plot([], [], 'b-', lw=2, label='Стержень OA')
line_OB, = ax_anim.plot([], [], 'r-', lw=2, label='Стержень OB')
spring, = ax_anim.plot([], [], 'g--', lw=1, label='Пружина')
point_O = Circle((0, 0), 0.05, fc='k')
ax_anim.add_patch(point_O)
ax_anim.legend()

def update(frame):
    phi1 = phi1_sol[frame]
    phi2 = phi2_sol[frame]
    xA, yA = l1*np.cos(phi1), l1*np.sin(phi1)
    xB, yB = l2*np.cos(phi2), l2*np.sin(phi2)
    xD, yD = a*np.cos(phi1), a*np.sin(phi1)
    xE, yE = a*np.cos(phi2), a*np.sin(phi2)
    line_OA.set_data([0, xA], [0, yA])
    line_OB.set_data([0, xB], [0, yB])
    spring.set_data([xD, xE], [yD, yE])
    return line_OA, line_OB, spring

anim = FuncAnimation(fig_anim, update, frames=len(t), interval=20, blit=True)

fig_plots, (ax_angles, ax_reactions) = plt.subplots(2, 1, figsize=(10, 8))
ax_angles.plot(t, phi1_sol, label='φ1')
ax_angles.plot(t, phi2_sol, label='φ2')
ax_angles.set_xlabel('Время (с)')
ax_angles.set_ylabel('Угол (рад)')
ax_angles.set_title('Угловые положения')
ax_angles.grid(True)
ax_angles.legend()
ax_reactions.plot(t, Rx, label='Rx')
ax_reactions.plot(t, Ry, label='Ry')
ax_reactions.set_xlabel('Время (с)')
ax_reactions.set_ylabel('Сила реакции (Н)')
ax_reactions.set_title('Силы реакции')
ax_reactions.grid(True)
ax_reactions.legend()
plt.tight_layout()
plt.show()