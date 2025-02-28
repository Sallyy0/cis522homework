import gymnasium as gym
import numpy as np


class Agent:
    """
    Agent Class for creating the Agent
    """

    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        """
        function that creates an intial Agent object
        """
        self.action_space = action_space
        self.observation_space = observation_space
        self.bins = [
            np.linspace(observation_space.low[i], observation_space.high[i], 20)
            for i in range(self.observation_space.shape[0])
        ]
        self.q_table = np.ones((*[len(b) + 1 for b in self.bins], self.action_space.n))
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 0.03

    def discretize(self, observation):
        """
        return discrete observations and bins
        """
        return tuple(np.digitize(obs, bin) for obs, bin in zip(observation, self.bins))

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        """
        find the proper action under the policy
        """
        discrete_obs = self.discretize(observation)
        if np.random.uniform(0, 1) < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.q_table[discrete_obs])

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """
        update the policy
        """
        discrete_obs = self.discretize(observation)
        if terminated or truncated:
            self.q_table[discrete_obs] = reward
        else:
            max_future_q = np.max(self.q_table[discrete_obs])
            current_q = self.q_table[discrete_obs]
            new_q = (1 - self.alpha) * current_q + self.alpha * (
                reward + self.gamma * max_future_q
            )
            self.q_table[discrete_obs] = new_q
