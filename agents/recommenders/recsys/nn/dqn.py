from torch import FloatTensor, max, LongTensor, BoolTensor, gather, Tensor
from numpy import array, number, ravel
from torch.nn import Sequential, Linear, ReLU, MSELoss, Module
from torch.optim import Adam
from typing import List, Any, Tuple, Optional

from traitlets import Int
from agents.recommenders.recsys.nn.base_network import BaseNetwork


# def sequential_architecture(observation_space: int, action_space: int, num_hidden: int, bias: bool = True) -> Module:
#     """ Fully connected layers, with bias, and ReLU activation"""
#     network = Sequential(
#         Linear(observation_space, num_hidden, bias),
#         ReLU(),
#         Linear(num_hidden, num_hidden),
#         ReLU(),
#     )
#     head = Linear(num_hidden, action_space)
#     return network

def sequential_architecture(layers: List[int], bias: bool = True) -> Module:
    """ Fully connected layers, with bias, and ReLU activation"""
    architecture = []
    for i in range(len(layers) - 2):
        architecture.append(Linear(layers[i], layers[i + 1]))
        architecture.append(ReLU())
    architecture.append(Linear(layers[-2], layers[-1]))
    return Sequential(*architecture)


class DeepQNetwork(BaseNetwork):
    """Implementation of a Deep Q Network with a Sequential arquitecture. Layers are
    supposed to be provided as a list of torch modules."""

    def __init__(
        self,
        learning_rate: float,
        architecture: Module,
        discount_factor: float = 0.99,
    ):
        super().__init__()
        self.model = architecture
        self.discount_factor = discount_factor
        self.optimizer = Adam(self.parameters(), lr=learning_rate)
        if self.device == "cuda":
            self.model.cuda()

    def best_action_for_state(self, state: Any) -> Any:
        if type(state) is tuple:
            state = array([ravel(s) for s in state])
        state_tensor = FloatTensor(state).to(device=self.device)
        q_values = self.model(state_tensor)
        best_action = max(q_values, dim=-1)[1].item()
        return best_action

    def learn_from(self, experiences: List[Tuple]):
        self.optimizer.zero_grad()
        loss = self._calculate_loss(experiences)
        loss.backward()
        self.optimizer.step()

    def _calculate_loss(self, experiences: List[Tuple]) -> Tensor:
        states, actions, rewards, dones, next_states = [i for i in experiences]
        state_tensors = FloatTensor(states).to(device=self.device)
        next_state_tensors = FloatTensor(next_states).to(device=self.device)
        reward_tensors = FloatTensor(rewards).to(device=self.device).reshape(-1, 1)
        action_tensors = (
            LongTensor(array(actions)).reshape(-1, 1).to(device=self.device)
        )
        done_tensors = BoolTensor(dones).to(device=self.device)
        actions_for_states = self.model(state_tensors)
        q_vals = gather(actions_for_states, 1, action_tensors)
        next_actions = [self.best_action_for_state(s) for s in next_states]
        next_action_tensors = (
            LongTensor(next_actions).reshape(-1, 1).to(device=self.device)
        )
        q_vals_next = gather(self.model(next_state_tensors), 1, next_action_tensors)
        q_vals_next[done_tensors] = 0
        expected_q_vals = self.discount_factor * q_vals_next + reward_tensors
        return MSELoss()(q_vals, expected_q_vals.reshape(-1, 1))