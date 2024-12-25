import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow

t = np.linspace(0, 10, 1000)
dt = t[1] - t[0]

r = 1 + 1.5 * np.sin(12 * t)
phi = 1.2 * t + 0.2 * np.cos(12 * t)

x = r * np.cos(phi)
y = r * np.sin(phi)

vx = np.gradient(x, dt)
vy = np.gradient(y, dt)

acc_x = np.gradient(vx, dt)
acc_y = np.gradient(vy, dt)

speed = np.sqrt(vx**2 + vy**2)
tangential_acc = np.gradient(speed, dt)

v_magnitude = np.sqrt(vx**2 + vy**2)
v_unit_x = vx / v_magnitude
v_unit_y = vy / v_magnitude

tang_acc_x = tangential_acc * v_unit_x
tang_acc_y = tangential_acc * v_unit_y

fig, axis = plt.subplots(figsize=(10, 10))
axis.set_aspect('equal')
axis.grid(True)

axis.plot(x, y, 'b-', alpha=0.3, label='Trajectory')

point, = axis.plot([], [], 'ro', markersize=10, label='Point')

padding = 1.2
axis.set_xlim(np.min(x) * padding, np.max(x) * padding)
axis.set_ylim(np.min(y) * padding, np.max(y) * padding)

vel_arrow = None
acc_arrow = None
tang_acc_arrow = None

def update(frame):
    global vel_arrow, acc_arrow, tang_acc_arrow

    if vel_arrow:
        vel_arrow.remove()
    if acc_arrow:
        acc_arrow.remove()
    if tang_acc_arrow:
        tang_acc_arrow.remove()

    point.set_data([x[frame]], [y[frame]])

    vel_scale = 0.05
    vel_dx = vx[frame] * vel_scale
    vel_dy = vy[frame] * vel_scale
    vel_arrow = axis.arrow(x[frame], y[frame], vel_dx, vel_dy,
                          head_width=0.1, head_length=0.2, fc='g', ec='g',
                          label='Velocity')

    acc_scale = 0.005
    acc_dx = acc_x[frame] * acc_scale
    acc_dy = acc_y[frame] * acc_scale
    acc_arrow = axis.arrow(x[frame], y[frame], acc_dx, acc_dy,
                          head_width=0.1, head_length=0.2, fc='r', ec='r',
                          label='Acceleration')

    tang_acc_scale = 0.005
    tang_acc_dx = tang_acc_x[frame] * tang_acc_scale
    tang_acc_dy = tang_acc_y[frame] * tang_acc_scale
    tang_acc_arrow = axis.arrow(x[frame], y[frame], tang_acc_dx, tang_acc_dy,
                               head_width=0.1, head_length=0.2, fc='blue', ec='blue',
                               label='Tangential Acceleration')

    return point, vel_arrow, acc_arrow, tang_acc_arrow

anim = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)

axis.legend()
axis.set_title('Point Motion with Velocity and Acceleration Vectors')

plt.show()