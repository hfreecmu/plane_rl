import os
import gym
import gym_plane
from plane_toolkit.agents import agent_dict
import copy
import numpy as np
import matplotlib.pyplot as plt


def create_info_dict(**kwargs):
    info_dict = {}
    info_dict["target_traj_yml"] = kwargs["target_traj_yml"]
    info_dict["pos_rew_scale"] = kwargs["pos_rew_scale"]
    info_dict["quat_rew_scale"] = kwargs["quat_rew_scale"]
    info_dict["u_rew_scale"] = kwargs["u_rew_scale"]
    info_dict["obs_bound_factor"] = kwargs["obs_bound_factor"]

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

def plot_all_rewards(full_rewards, agent_names, reward_file, tsamp, tag="Reward"):
    means = []
    stds = []
    full_rewards = np.array(full_rewards)
    for i in range(full_rewards.shape[1]):
        agent_rewards = full_rewards[:, i, :]
        means.append(np.mean(agent_rewards, axis=0))
        stds.append(np.std(agent_rewards, axis=0))
    for agent_name, mean, std in zip(agent_names, means, stds):
        plt.plot(tsamp[0:mean.shape[0]], mean, label=f"{tag} {agent_name}")
        plt.fill_between(tsamp[0:mean.shape[0]], mean - std, mean + std, alpha=0.3)
    plt.legend()
    plt.savefig(reward_file)
    plt.clf()
    plt.close()

def plot_reward(rewards, agent_name, reward_file, tsamp, tag="Reward"):
    y = np.array(rewards)
    plt.plot(tsamp[0:y.shape[0]], y)
    plt.ylabel(tag)
    plt.xlabel("Time (s)")
    title = f"Performance of {agent_name} Agent"
    plt.title(title)
    plt.savefig(reward_file)

    plt.clf()

def plot_all_trajectories(full_positions, agent_names, trajectory_file, tsamp):
    means = []
    full_positions = np.array(full_positions)
    ax = plt.subplot(projection='3d')
    for i in range(full_positions.shape[1]):
        agent_positions = full_positions[:, i, :]
        means.append(np.mean(agent_positions, axis=0))
    for agent_name, mean in zip(agent_names, means):
        ax.scatter(mean[:, 0], mean[:, 1], mean[:, 2], label=f"Trajectory of {agent_name}")
    plt.legend()
    plt.savefig(trajectory_file)
    plt.clf()
    plt.close()

def save_and_plot_trajctory(positions, agent_name, 
                            trajectory_file, trajectory_vis_file, 
                            tsamp):
    positions = np.array(positions)
    ax = plt.subplot(projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
    title = f"Trajectory of {agent_name} Agent"
    plt.savefig(trajectory_vis_file)

    plt.clf()

    np.savetxt(trajectory_file, positions, delimiter=',')

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
    reward_files = []
    pos_error_files = []
    quat_error_files = []
    u_error_files = []
    trajectory_files = []
    trajectory_vis_files = []
    for agent_type_t in agent_types:
        vis_dir_agent_pre_trial = os.path.join(vis_dir, agent_type_t)
        if not os.path.exists(vis_dir_agent_pre_trial):
            os.mkdir(vis_dir_agent_pre_trial)

        vis_dir_agent = os.path.join(vis_dir_agent_pre_trial, "trial_" + str(trial_num))
        if not os.path.exists(vis_dir_agent):
            os.mkdir(vis_dir_agent)

        vis_dirs.append(vis_dir_agent)

        model_dirs.append(os.path.join(model_dir, agent_type_t))

        reward_files.append(os.path.join(vis_dir_agent, "reward.png"))
        pos_error_files.append(os.path.join(vis_dir_agent, "pos_error.png"))
        quat_error_files.append(os.path.join(vis_dir_agent, "quat_error.png"))
        u_error_files.append(os.path.join(vis_dir_agent, "u_error.png"))
        trajectory_files.append(os.path.join(vis_dir_agent, "trajectory.csv"))
        trajectory_vis_files.append(os.path.join(vis_dir_agent, "trajectory_vis.png"))

    # Merge kwargs and locals appropriately
    kwargs.update(locals())
    kwargs.pop("kwargs")
    info_dict = create_info_dict(**kwargs)

    envs = [None] * len(agent_types)
    envs[0] = gym.make("plane-v0", info_dict=info_dict)

    tsamp = np.arange(0, envs[0].Tf+ envs[0].h, envs[0].h)

    dones = [False] * len(agent_types)
    positions = [None] * len(agent_types)
    rewards = [None] * len(agent_types)
    pos_errors = [None] * len(agent_types)
    quat_errors = [None] * len(agent_types)
    u_errors = [None] * len(agent_types)

    obs = [None] * len(agent_types)
    obs[0] = envs[0].reset()

    for i in range(0, len(agent_types)):
        if i > 0:
            envs[i] = copy.deepcopy(envs[0])
            obs[i] = copy.deepcopy(obs[0])

        positions[i] = []
        #positions[i].append(obs[i][0:3])
        positions[i].append(envs[i].x[0:3])

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
            obs[i], reward, dones[i], _ = envs[i].step(action)
            pos_error = envs[i].latest_pos_error
            quat_error = envs[i].latest_quat_error
            u_error = envs[i].latest_u_error

            if rewards[i] is None:
                rewards[i] = []

            if pos_errors[i] is None:
                pos_errors[i] = []

            if quat_errors[i] is None:
                quat_errors[i] = []

            if u_errors[i] is None:
                u_errors[i] = []

            #positions[i].append(obs[i][0:3])
            positions[i].append(envs[i].x[0:3])
            rewards[i].append(reward)
            pos_errors[i].append(pos_error)
            quat_errors[i].append(quat_error)
            u_errors[i].append(u_error)

    for i in range(len(agent_types)):
        plot_reward(rewards[i], agents[i].get_name(), reward_files[i], tsamp[1:])
        plot_reward(pos_errors[i], agents[i].get_name(), pos_error_files[i], tsamp[1:], tag="Position Erros")
        plot_reward(quat_errors[i], agents[i].get_name(), quat_error_files[i], tsamp[1:], tag="Quat Errors")
        plot_reward(u_errors[i], agents[i].get_name(), u_error_files[i], tsamp[1:], tag="U Errors")
        save_and_plot_trajctory(positions[i], agents[i].get_name(), 
                                trajectory_files[i], trajectory_vis_files[i],
                                tsamp)
        

    return positions, rewards, pos_errors, quat_errors, tsamp

def test_agents(
    agent_types, num_trials, vis_dir, **kwargs,  # Unused, for compatability
):
    kwargs.update(locals())
    kwargs.pop("kwargs")

    full_positions = []
    full_rewards = []
    full_pos_errors = []
    full_quat_errors = []
    full_tsamps = []
    for trial_num in range(num_trials):
        positions, rewards, pos_errors, quat_errors, tsamp = run_trial(trial_num=trial_num, **kwargs)
        full_positions.append(positions)
        full_rewards.append(rewards)
        full_pos_errors.append(pos_errors)
        full_quat_errors.append(quat_errors)
        full_tsamps.append(tsamp)

    reward_comparison_file = os.path.join(vis_dir, "reward_comparison.png")
    plot_all_rewards(full_rewards, agent_types, reward_comparison_file, tsamp[1:])

    pos_error_comparison_file = os.path.join(vis_dir, "pos_errors_comparison.png")
    plot_all_rewards(full_pos_errors, agent_types, pos_error_comparison_file, tsamp[1:], tag="Pos error")
    
    quat_error_comparison_file = os.path.join(vis_dir, "quat_errors_comparison.png")
    plot_all_rewards(full_quat_errors, agent_types, quat_error_comparison_file, tsamp[1:], tag="Quat error")
    
    trajectory_comparison_file = os.path.join(vis_dir, "position_comparison.png")
    plot_all_trajectories(full_positions, agent_types, trajectory_comparison_file, tsamp)

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

    kwargs.update(locals())
    kwargs.pop("kwargs")
    info_dict = create_info_dict(**kwargs)

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