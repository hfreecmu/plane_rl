import gym
import gym_plane
from stable_baselines3.common.env_checker import check_env
from pandas import read_csv

def run():
    info_dict = {}
    info_dict["target_traj_yml"] = '/home/frc-ag-3/harry_ws/courses/grad_ai/final_project/trajectories/iLQR_loop/iqrl.yml'
    info_dict["obs_bound_factor"] = 2.0
    info_dict["pos_tol"] = 1.0

    env = gym.make("plane-v0", info_dict=info_dict)
    check_env(env)

    print('env check passed')

if __name__ == "__main__":
    run()