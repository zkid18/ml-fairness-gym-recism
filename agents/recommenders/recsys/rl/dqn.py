from numpy.random import RandomState
from abc import ABC, abstractmethod
from typing import List, Any
from numpy import arange
from agents.recommenders.recsys.rl.base_agent import ReinforcementLearning

from agents.recommenders.recsys.experience_replay.experience_buffer import (
    ExperienceReplayBuffer, 
    ExperienceReplayBufferParameters
)

from agents.recommenders.recsys.nn.dqn import DeepQNetwork, sequential_architecture


class DecayingEpsilonGreedy(ReinforcementLearning, ABC):
    def __init__(
        self,
        initial_exploration_probability: float = 0.2,
        decay_rate: float = 1,
        minimum_exploration_probability=0.01,
        random_state: RandomState = RandomState(),
    ):
        self.random_state = random_state
        self.epsilon = initial_exploration_probability
        self.minimum_exploration_probability = minimum_exploration_probability
        self.decay_rate = decay_rate

    def action_for_state(self, state: Any) -> Any:
        """With probability epsilon, we explore by sampling one of the random available actions.
        Otherwise we exploit by chosing the action with the highest Q value."""
        if self.random_state.random() < self.epsilon:
            action = self.explore()
        else:
            action = self.exploit(state)
        return action

    def _decay(self):
        """ Slowly decrease the exploration probability. """
        self.epsilon = max(
            self.epsilon * self.decay_rate, self.minimum_exploration_probability
        )

    @abstractmethod
    def explore(self) -> Any:
        """ Randomly selects an action"""
        pass

    @abstractmethod
    def exploit(self, state: Any) -> Any:
        """ Selects the best action known for the given state """
        pass


class DQNAgent(DecayingEpsilonGreedy):
    """ TODO: This agent needs to be fixed"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List,
        network_update_frequency: int = 3,
        initial_exploration_probability: float = 1.0,
        decay_rate: float = 0.99,
        minimum_exploration_probability=0.05,
        buffer_size: int = 10000,
        buffer_burn_in: int = 1000,
        batch_size: int = 32,
        discount_factor: float = 0.99,
        learning_rate: float = 0.99,
        random_state: RandomState = RandomState(seed=42),
    ):

        architecture = sequential_architecture(
            [input_size] + hidden_layers + [output_size]
        )
        self.network = DeepQNetwork(learning_rate, architecture, discount_factor)
        self.buffer = ExperienceReplayBuffer(
            ExperienceReplayBufferParameters(
                max_experiences=buffer_size,
                minimum_experiences_to_start_predicting=buffer_burn_in,
                batch_size=batch_size,
                random_state=random_state,
            )
        )

        super().__init__(
            initial_exploration_probability,
            decay_rate,
            minimum_exploration_probability,
            random_state,
        )

        self.step_count = 0
        self.network_update_frequency = network_update_frequency
        self.actions = arange(output_size)


    def _check_update_network(self):
        if self.buffer.ready_to_predict():
            self.step_count += 1
            if self.step_count == self.network_update_frequency:
                self.step_count = 0
                batch = self.buffer.sample_batch()
                self.network.learn_from(batch)


    def action_for_state(self, state: Any) -> Any:
        state_flat = state.flatten()
        if self.buffer.ready_to_predict():
            action = super().action_for_state(state_flat)
        else:
            action = self.explore()
        self._check_update_network()
        return action


    def top_k_actions_for_state(self, state: Any, k: int = 1) -> Any:
        # TODO:
        pass


    def explore(self):
        return self.random_state.choice(self.actions)


    def exploit(self, state: Any):
        return self.network.best_action_for_state(state)

    def _decay(self):
        """ Slowly decrease the exploration probability. """
        self.epsilon = max(
            self.epsilon * self.decay_rate, self.minimum_exploration_probability
        ) 

    def store_experience(
        self, state: Any, action: Any, reward: float, done: bool, new_state: Any
    ):
        if done and self.buffer.ready_to_predict():
            self._decay()
        state_flat = state.flatten()
        new_state_flat = new_state.flatten()
        self.buffer.store_experience(state_flat, action, reward, done, new_state_flat)