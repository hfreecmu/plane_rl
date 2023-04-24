from plane_toolkit.agents.StableBaselinesAgent import (
    PPOAgent,
    TD3Agent,
)
from plane_toolkit.agents.MBAgent import MBAgent
from plane_toolkit.agents.BehaviorCloningAgent import BehaviorCloningAgent
from plane_toolkit.agents.ConvexOptAgent import ConvexOptAgent

agent_dict = {
    "PPO": PPOAgent,
    "TD3": TD3Agent,
    "MB": MBAgent,
    "BC": BehaviorCloningAgent,
    "CO": ConvexOptAgent,
}
