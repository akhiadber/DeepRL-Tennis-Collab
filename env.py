"""
UnityML Environment.
"""

from unityagents import UnityEnvironment

class UnityMultiAgent():
    """Multi-agent UnityML environment."""

    def __init__(self, evaluation_only=False, seed=0, file_name='Tennis_Linux_NoVis/Tennis.x86_64'):
        """Load env file (platform specific, see README) and initialize the environment."""        
        self.env = UnityEnvironment(file_name=file_name, seed=seed)
        self.brain_name = self.env.brain_names[0]
        self.evaluation_only = evaluation_only

    def reset(self):
        """Reset the environment."""
        info = self.env.reset(train_mode=not self.evaluation_only)[self.brain_name]
        state = info.vector_observations
        return state

    def step(self, action):
        """Take a step in the environment."""
        info = self.env.step(action)[self.brain_name]
        state = info.vector_observations
        reward = info.rewards
        done = info.local_done
        return state, reward, done
