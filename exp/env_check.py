import gym
import gym_plane
from stable_baselines3.common.env_checker import check_env
from pandas import read_csv

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
    info_dict["is_debug"] = False

    #load target_traj
    target_traj_path = '/home/frc-ag-3/harry_ws/courses/grad_ai/final_project/harry_traj.csv'
    df = read_csv(target_traj_path, header=None)
    target_traj = df.values
    info_dict["target_traj"] = target_traj

    env = gym.make("plane-v0", info_dict=info_dict)
    check_env(env)

if __name__ == "__main__":
    run()