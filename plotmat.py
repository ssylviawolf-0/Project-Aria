import numpy as np
import matplotlib.pyplot as plt

# 1. Load data from CSV
# File expected to have 4 columns and N rows (where N is a multiple of 4)
raw_data = np.loadtxt('./files/Demonstrations/aria_project/ee_trajectory.csv', delimiter=',', skiprows=1)
from matplotlib.animation import FuncAnimation

# 1. Load data and skip the first header row

# 2. Reshape into 4x4 matrices
matrices = raw_data.reshape(-1, 4, 4)

# 3. Compute coordinates sequentially over time
start_point = np.array([1.0, 1.0, 1.0, 1.0])  # Adjust your initial point [X, Y, Z, 1]
points = [start_point]

current_point = start_point
for M in matrices:
    current_point = np.dot(M, current_point)
    points.append(current_point)

points = np.array(points)  # Converts to shape (Num_Steps + 1, 4)
x_vals = points[:, 0]
y_vals = points[:, 1]
z_vals = points[:, 2]

# 4. Set up the 3D Plot environment
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

# Set fixed axes limits based on your data spread so the viewport doesn't jump
ax.set_xlim([np.min(x_vals) - 1, np.max(x_vals) + 1])
ax.set_ylim([np.min(y_vals) - 1, np.max(y_vals) + 1])
ax.set_zlim([np.min(z_vals) - 1, np.max(z_vals) + 1])

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
title = ax.set_title('Transformation Trajectory (Time: 0)')

# Initialize empty visual elements to update during animation
line, = ax.plot([], [], [], color='blue', linestyle='--', alpha=0.6)
scatter = ax.scatter([], [], [], color='red', s=40)
start_marker = ax.scatter([x_vals[0]], [y_vals[0]], [z_vals[0]], color='green', s=100, marker='*')

# 5. Define animation update function (called for each frame/timestep)
def update(frame):
    # Slice data up to the current time frame
    line.set_data(x_vals[:frame+1], y_vals[:frame+1])
    line.set_3d_properties(z_vals[:frame+1])
    
    # Update scatter dots positions
    scatter._offsets3d = (x_vals[:frame+1], y_vals[:frame+1], z_vals[:frame+1])
    
    # Update time title
    title.set_text(f'Transformation Trajectory (Time Step: {frame})')
    return line, scatter

# 6. Run and display the animation
# interval=500 means 500 milliseconds (0.5 seconds) per frame
ani = FuncAnimation(fig, update, frames=len(x_vals), interval=500, blit=False, repeat=False)

plt.show()
