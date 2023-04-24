from plane_toolkit.agents.StableBaselinesAgent import (
    PPOAgent,
    TD3Agent,
)
from plane_toolkit.agents.MBAgent import MBAgent
from plane_toolkit.agents.BehaviorCloningAgent import BehaviorCloningAgent

agent_dict = {
    "PPO": PPOAgent,
    "TD3": TD3Agent,
    "MB": MBAgent,
    "BC": BehaviorCloningAgent,
}
