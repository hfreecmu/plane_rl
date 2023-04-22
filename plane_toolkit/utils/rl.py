import os
import pandas as pd
import gym
import gym_plane
from plane_toolkit.agents import agent_dict
import copy
import numpy as np
import matplotlib.pyplot as plt


def create_info_dict():
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
    target_traj_path = '/home/frc-ag-3/harry_ws/courses/grad_ai/final_project/loop_ref.csv'
    df = pd.read_csv(target_traj_path, header=None)
    target_traj = df.values
    info_dict["target_traj"] = target_traj

    return info_dict

def build_train_cfg(
    num_par,
    learning_rate,
    n_steps,
    total_timesteps,
    verbose,
    save_freq,
    model_dir,
    log_dir,
):
    cfg = {
        "num_par": num_par,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "total_timesteps": total_timesteps,
        "verbose": verbose,
        "save_freq": save_freq,
        "model_dir": model_dir,
        "log_dir": log_dir,
    }

    return cfg

def run_trial(
    agent_types,
    policy,
    vis_dir,
    trial_num,
    model_dir,
    _run,
    **kwargs,
):
    if len(agent_types) == 0:
        raise RuntimeError("More than one agent_type required")

    for agent_type_t in agent_types:
        assert agent_type_t in agent_dict

    vis_dirs = []
    model_dirs = []
    for agent_type_t in agent_types:
        vis_dir_agent_pre_trial = os.path.join(vis_dir, agent_type_t)
        if not os.path.exists(vis_dir_agent_pre_trial):
            os.mkdir(vis_dir_agent_pre_trial)

        vis_dir_agent = os.path.join(vis_dir_agent_pre_trial, "trial_" + str(trial_num))
        if not os.path.exists(vis_dir_agent):
            os.mkdir(vis_dir_agent)

        vis_dirs.append(vis_dir_agent)

        model_dirs.append(os.path.join(model_dir, agent_type_t))

    # Merge kwargs and locals appropriately
    #kwargs.update(locals())
    #kwargs.pop("kwargs")
    #info_dict = create_info_dict(**kwargs)
    info_dict = create_info_dict()

    envs = [None] * len(agent_types)
    envs[0] = gym.make("plane-v0", info_dict=info_dict)

    dones = [False] * len(agent_types)
    positions = [None] * len(agent_types)

    obs = [None] * len(agent_types)
    obs[0] = envs[0].reset()

    for i in range(0, len(agent_types)):
        if i > 0:
            envs[i] = copy.deepcopy(envs[0])
            obs[i] = copy.deepcopy(obs[0])

    agents = []
    for i in range(len(agent_types)):
        agent = agent_dict[agent_types[i]](envs[i])
        agent.policy = policy
        agent.load_model(model_dirs[i])
        agents.append(agent)

    while (np.sum(dones) < len(agent_types)):
        for i in range(len(agent_types)):
            if dones[i]:
                continue

            action, _ = agents[i].get_action(obs[i], envs[i])
            obs[i], _, dones[i], _ = envs[i].step(action)

            if positions[i] is None:
                positions[i] = []

            positions[i].append(obs[i][0:3])

    return positions

def test_agents(
    agent_types, num_trials, vis_dir, **kwargs,  # Unused, for compatability
):
    kwargs.update(locals())
    kwargs.pop("kwargs")

    full_positions = []
    for trial_num in range(num_trials):
        positions = run_trial(trial_num=trial_num, **kwargs)
        full_positions.append(positions)

    test_pos = np.array(full_positions[0][0])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(test_pos[:, 0], test_pos[:, 1], test_pos[:, 2])
    plt.show()

def train_agent(
    agent_type,
    policy,
    model_dir,
    log_dir,
    num_par,
    learning_rate,
    n_steps,
    total_timesteps,
    verbose,
    save_freq,
    **kwargs,
):
    model_dir = os.path.join(model_dir, agent_type)
    log_dir = os.path.join(log_dir, agent_type)

    #kwargs.update(locals())
    #kwargs.pop("kwargs")
    #info_dict = create_info_dict(**kwargs)
    info_dict = create_info_dict()

    env = gym.make("plane-v0", info_dict=info_dict)
    agent = agent_dict[agent_type](env.action_space)
    agent.policy = policy

    cfg = build_train_cfg(
        num_par,
        learning_rate,
        n_steps,
        total_timesteps,
        verbose,
        save_freq,
        model_dir,
        log_dir,
    )

    agent.train(env, cfg)
    return agent