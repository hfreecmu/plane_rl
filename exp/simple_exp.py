import gym
import gym_plane
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np

def run():
    info_dict = {}
    info_dict["Tf"] = 5.0
    info_dict["h"] = 1.0/20.0
    info_dict["num_prev_steps"] = 5
    info_dict["num_future_steps"] = 5
    info_dict["clip_observations"] = 100.0
    info_dict["clip_actions"] = 100.0
    info_dict["pos_rew_scale"] = 0.7
    info_dict["quat_rew_scale"] = 0.3
    info_dict["is_debug"] = True

    #load target_traj
    target_traj_path = '/home/frc-ag-3/harry_ws/courses/grad_ai/final_project/harry_traj.csv'
    df = read_csv(target_traj_path, header=None)
    target_traj = df.values
    info_dict["target_traj"] = target_traj

    env = gym.make("plane-v0", info_dict=info_dict)
    _ = env.reset()

    done = False
    safety_max = 1000
    safety_count = 0
    rewards = []
    positions = []
    while (not done) and (safety_count < safety_max):
        safety_count += 1

        random_action = env.action_space.sample()
        obs, reward, done, _ = env.step(random_action)
        positions.append(obs[0:3])
        rewards.append(reward)

    if safety_count == safety_max:
        raise RuntimeError("Safety limit reached")
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    positions = np.array(positions)
    rewards = np.array(rewards)

    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
    plt.show()

if __name__ == "__main__":
    run()