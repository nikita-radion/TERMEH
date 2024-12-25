import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import matplotlib.lines as lines

l1 = 1.25
l2 = 2.0
m1 = 1.0
m2 = 1.0
c = 5.0
a = 1
phi0 = np.pi/4

t = np.linspace(0, 10, 500)
dt = t[1] - t[0]

def phi1(t):
    return phi0/2 + 0.5 * np.sin(2*t)

def phi2(t):
    return phi0 + 0.5 * np.sin(1.5*t)

def get_positions(t):
    angle1 = phi1(t)
    angle2 = phi2(t)
    O = np.array([0, 0])
    A = np.array([l1 * np.cos(angle1), l1 * np.sin(angle1)])
    B = np.array([l2 * np.cos(angle2), l2 * np.sin(angle2)])
    D = O + (a/l1) * A
    E = O + (a/l2) * B
    return O, A, B, D, E

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.grid(True)
ax.set_aspect('equal')

line_OA, = ax.plot([], [], 'b-', lw=2, label='Rod OA')
line_OB, = ax.plot([], [], 'r-', lw=2, label='Rod OB')
spring, = ax.plot([], [], 'g--', lw=1, label='Spring')
point_O = Circle((0, 0), 0.1, fc='k')
ax.add_patch(point_O)

def update(frame):
    O, A, B, D, E = get_positions(t[frame])
    line_OA.set_data([O[0], A[0]], [O[1], A[1]])
    line_OB.set_data([O[0], B[0]], [O[1], B[1]])
    spring.set_data([D[0], E[0]], [D[1], E[1]])
    return line_OA, line_OB, spring

anim = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)
plt.legend()
plt.title('Two Rods Connected by Spring')
plt.show()