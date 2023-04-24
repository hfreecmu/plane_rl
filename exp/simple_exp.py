import gym
import gym_plane
from pandas import read_csv
import matplotlib.pyplot as plt
import numpy as np

def run():
    info_dict = {}
    info_dict["target_traj_yml"] = '/home/frc-ag-3/harry_ws/courses/grad_ai/final_project/trajectories/iLQR_loop/iqrl.yml'
    info_dict["obs_bound_factor"] = 2.0
    info_dict["pos_rew_scale"] = 1.0
    info_dict["quat_rew_scale"] = 1.0
    info_dict["u_rew_scale"] = 1.0

    env = gym.make("plane-v0", info_dict=info_dict)
    _ = env.reset()

    done = False
    safety_max = 1000
    safety_count = 0
    rewards = []
    positions = []
    while (not done) and (safety_count < safety_max):
        #random_action = env.action_space.sample()
        #obs, reward, done, _ = env.step(random_action)
        u = env.ilqr_control[safety_count]
        action = env.normalize_action(u)

        obs, reward, done, _ = env.step(action)
        pos = (obs[0:3]*env.pos_std) + env.pos_mean

        positions.append(pos)
        rewards.append(reward)

        safety_count += 1

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