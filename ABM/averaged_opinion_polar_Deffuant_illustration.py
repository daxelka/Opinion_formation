import math
import numpy as np
import matplotlib.pyplot as plt
import random
plt.rc('text', usetex=True)


def closest_average_angle(angle1, angle2):
    diff1 = abs(angle1 - angle2)
    diff2 = 2 * math.pi - abs(angle1 - angle2)
    if diff1 <= diff2:
        return min(angle1, angle2) + diff1 / 2
    else:
        if max(angle1, angle2) + diff2 / 2 < 2 * math.pi:
            return max(angle1, angle2) + diff2 / 2
        else:
            return max(angle1, angle2) + diff2 / 2 - 2 * math.pi


# angle1 = math.pi/5
# angle2 = math.pi - math.pi/3
angle1 = 2*math.pi*random.random()
angle2 = 2*math.pi*random.random()

result = closest_average_angle(angle1, angle2)
print(math.degrees(result))

angles = [angle1, angle2, result]
colors = ['black', 'black', '#1f77b4']

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4, 4))
# Plot origin (agent's start point)
ax.plot(0, 0, color='black', marker='o', markersize=5)

# Plot vectors
# for angle in angles:
for k, angle in enumerate(angles):
    u = np.cos(angle)
    v = np.sin(angle)
    # ax.quiver(0, 0, u, v, color=colors[k], angles="uv", scale=1.0)
    ax.axvline(x=angle, color=colors[k], alpha=1)


# Annotation
ax.annotate(r'$\phi_i^t$', xy=[angle1-0.3, 0.7], color='k', fontsize=20)
ax.annotate(r'$\phi_j^t$', xy=[angle2-0.3, 0.7], color='k', fontsize=20)
ax.annotate(r'$\phi_{i,j}^{t+1}$', xy=[result-0.3, 0.7], color='#1f77b4', fontsize=20)

# Plot configuration
ax.set_rticks([])
ax.set_rmin(0)
ax.set_rmax(1)
ax.set_thetalim(-np.pi, np.pi)
ax.set_xticks(np.linspace(np.pi, -np.pi, 4, endpoint=False))
ax.grid(True)
plt.show()
