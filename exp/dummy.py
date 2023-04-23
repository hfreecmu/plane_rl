import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

target_traj_path = '/home/frc-ag-3/harry_ws/courses/grad_ai/final_project/harry_traj.csv'
df = pd.read_csv(target_traj_path, header=None)
target_traj = df.values

positions = target_traj[:, 0:3]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
plt.show()