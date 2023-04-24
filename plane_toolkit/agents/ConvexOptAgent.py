from plane_toolkit.agents.BaseAgent import BaseAgent
import numpy as np
import gym

###WARNING WARNING WARNING
#modified line 634 in vim /home/frc-ag-3/.anaconda3/envs/ipp-toolkit/lib/python3.10/site-packages/imitation/algorithms/dagger.py
#I am not sure where this points in your env but it was required
#and I cannot determine where it has to be set when dagger is called
class ConvexOptAgent(BaseAgent):
    def __init__(self, env):
        self.name = "ConvexOpt"
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def get_name(self):
        return self.name

    def train(self, env, cfg):
        print("Cannot train convex opt agent.")

    def load_model(self, model_dir):
        pass

    def get_action(self, observation, env=None):
        curr_step = self.env.num_steps
        convex_controls = self.env.ilqr_control
        u = convex_controls[curr_step]
        action = env.normalize_action(u)
        return action, None
    
    def __call__(self, x):
        action, _ = self.get_action(x[0])
        return [action]
