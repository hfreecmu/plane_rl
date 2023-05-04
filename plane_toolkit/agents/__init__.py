from plane_toolkit.agents.StableBaselinesAgent import (
    DDPGAgent,
    PPOAgent,
    SACAgent,
    TD3Agent,
)
from plane_toolkit.agents.MBAgent import MBAgent
from plane_toolkit.agents.BehaviorCloningAgent import BehaviorCloningAgent
from plane_toolkit.agents.ConvexOptAgent import ConvexOptAgent

agent_dict = {
    "DDPG": DDPGAgent,
    "PPO": PPOAgent,
    "SAC": SACAgent,
    "TD3": TD3Agent,
    "MB": MBAgent,
    "BC": BehaviorCloningAgent,
    "CO": ConvexOptAgent,
}
