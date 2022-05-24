from numpy.random import RandomState
from abc import ABC, abstractmethod
from typing import List, Any
from copy import deepcopy
from numpy import arange
from agents.recommenders.recsys.rl.base_agent import ReinforcementLearning

from agents.recommenders.recsys.experience_replay.priority_experience_buffer import (
    PrioritizedExperienceReplayBuffer,
)
from agents.recommenders.recsys.experience_replay.experience_buffer import (
    PERBufferParameters,
    ExperienceReplayBufferParameters,
)

from agents.recommenders.recsys.statistics import LearningStatistics
from agents.recommenders.recsys.nn.dueling_dqn import DuelingDDQN


class RainbowDQNAgent(ReinforcementLearning):

    """Instead of sampling randomly from the buffer we prioritize experiences with PER
    Instead of epsilon-greedy we use gaussian noisy layers for exploration
    Instead of the Q value we calculate Value and Advantage (Dueling DQN).
    This implementation does not include the Categorical DQN part (yet)."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = None,
        network_update_frequency: int = 5,
        network_sync_frequency: int = 200,
        priority_importance: float = 0.6,
        priority_weigth_growth: float = 0.001,
        buffer_size: int = 10000,
        buffer_burn_in: int = 1000,
        batch_size: int = 32,
        noise_sigma: float = 0.017,
        discount_factor: float = 0.99,
        learning_rate: float = 0.0001,
        random_state: RandomState = RandomState(),
        statistics: LearningStatistics = None,
    ):

        self.network = DuelingDDQN(
            n_input=input_size,
            n_output=output_size,
            learning_rate=learning_rate,
            noise_sigma=noise_sigma,
            discount_factor=discount_factor,
            statistics=statistics,
            hidden_layers=hidden_layers,
        )
        self.target_network = deepcopy(self.network)

        self.buffer = PrioritizedExperienceReplayBuffer(
            ExperienceReplayBufferParameters(
                max_experiences=buffer_size,
                minimum_experiences_to_start_predicting=buffer_burn_in,
                batch_size=batch_size,
                random_state=random_state,
            ),
            PERBufferParameters(
                alpha=priority_importance,
                beta_growth=priority_weigth_growth,
            ),
        )
        self.step_count = 0
        self.network_update_frequency = network_update_frequency
        self.network_sync_frequency = network_sync_frequency
        self.actions = arange(output_size)
        self.random_state = random_state

    def _check_update_network(self):
        # we only start training the network once the buffer is ready
        # (the burn in is filled)
        if self.buffer.ready_to_predict():
            self.step_count += 1
            if self.step_count % self.network_update_frequency == 0:
                # we train at every K steps
                self.network.learn_with(self.buffer, self.target_network)
            if self.step_count % self.network_sync_frequency == 0:
                # at every N steps replaces the target network with the main network
                self.target_network.load_state_dict(self.network.state_dict())

    def top_k_actions_for_state(self, state: Any, k: int = 1) -> Any:
        state_flat = state.flatten()
        if self.buffer.ready_to_predict():
            actions = self.target_network.top_k_actions_for_state(state_flat, k=k)
        else:
            actions = self.random_state.choice(self.actions, size=k)
        self._check_update_network()
        return actions

    def action_for_state(self, state: Any) -> Any:
        return self.top_k_actions_for_state(state, k=1)[0]

    def store_experience(
        self, state: Any, action: Any, reward: float, done: bool, new_state: Any
    ):
        state_flat = state.flatten()
        new_state_flat = new_state.flatten()
        self.buffer.store_experience(state_flat, action, reward, done, new_state_flat)