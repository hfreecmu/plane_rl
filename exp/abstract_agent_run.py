# Needed for env instantiation
from sacred import Experiment

from plane_toolkit.utils.rl import train_agent, test_agents

ex = Experiment("rl_train_test")

@ex.config
def config():
    agent_types = ["CO"]  # Which agents to train or test on
    policy = "MlpPolicy"  # What policy to use, can also be CNN
    num_trials = 1#20  # How many test runs to run
    vis_dir = "vis"  # Where to save visualization
    model_dir = "models"  # Where to save and/or load models
    log_dir = "logs"
    num_par = 1
    # learning_rate can be set on the command line
    LR_DICT = {
        "DQN": 0.0001,
        "PPO": 1e-3,#0.0003,
        "DDPG": 0.001,
        "SAC": 0.0003,
        "random": None,
        "MB": 0.0005,
        "UCB": None,
        "BC": None,
        "DA": None,
        "Perfect": None,
        "TD3": 0.0005,
        "CO": None,
    }
    learning_rate = LR_DICT[agent_types[0]]
    n_steps = 4096#2048
    total_timesteps = 1000000#300000
    verbose = 1
    save_freq = 1000
    pos_rew_scale = 1.0
    quat_rew_scale = 1.0
    u_rew_scale = 1.0
    #target_traj_yml = '/home/frc-ag-3/harry_ws/courses/grad_ai/final_project/trajectories/iLQR_loop/iqrl.yml'
    target_traj_yml = '/home/frc-ag-3/harry_ws/courses/grad_ai/final_project/trajectories/iLQR_immelman/iqrl.yml'
    obs_bound_factor = 10.0
    pos_tol = 1.0
    train = False

@ex.automain
def main(
    agent_types,
    policy,
    num_trials,
    vis_dir,
    model_dir,
    log_dir,
    num_par,
    learning_rate,
    n_steps,
    total_timesteps,
    verbose,
    save_freq,
    pos_rew_scale,
    quat_rew_scale,
    u_rew_scale,
    target_traj_yml,
    obs_bound_factor,
    pos_tol,
    train,
    _run,
):
    if train:
        train_agent(agent_type=agent_types[0], **locals())
    else:
        test_agents(**locals())
