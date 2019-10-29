'''
Main background for this found here:
    https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
'''
import sys
import math
import time
import torch
import random
import numpy as np
import torch.optim as optim
from game import Game
from utils_imgs import *
from itertools import count
from collections import namedtuple


class DQN(torch.nn.Module):
    '''
    This is our custom model that we are building.  For now, it is copied
    from the tutorial.
    '''

    def __init__(self, hidden_dim=64):
        super(DQN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(5, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 8)
        )

    def forward(self, x):
        '''
        This outlines how the data flows through this NN.
        '''
        out = self.layer1(x)
        return out


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def optimize_model(optimizer, policy_net, target_net, memory,
                   BATCH_SIZE=128, GAMMA=0.999):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(
        lambda s: s is not None,
        batch.next_state
    )), dtype=torch.long)
    non_final_next_states = torch.cat([
        s for s in batch.next_state
        if s is not None
    ])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the
    # expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    # print("\t", non_final_next_states.shape)
    next_state_values[non_final_mask] =\
        target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = torch.nn.functional.smooth_l1_loss(
        state_action_values,
        expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def parse_action(action):
    return torch.tensor(
        [np.nanargmax(action.detach().numpy())],
        dtype=torch.long
    )


def select_action(t, state, policy_net,
                  EPS_START=0.9,
                  EPS_END=0.05,
                  EPS_DECAY=200.0):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * t / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(8)]], dtype=torch.long)


def train_model(N_games=10,
                max_game_loop=1000,
                learning_rate=1e-4,
                TARGET_UPDATE=2):
    '''
    This will train the NN.
    '''
    g = Game()
    policy_net = DQN()
    target_net = DQN()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())

    # Unsure if I want to use this or not.  For now, skip
    memory = ReplayMemory(10000)

    sys.stdout.write("Training...\n")
    sys.stdout.flush()

    for i in range(N_games):
        # Start a new "game"
        g.reset()

        state = g.get_state()

        sys.stdout.write("\r|" + "=" * i + "-" * (N_games - i) + "|")
        sys.stdout.flush()
        for t in count():
            # Select and perform an action
            action = select_action(t, state, policy_net)
            reward, done = g.step(action.item())

            # Store the transition in memory
            if g.is_finished() or g.timed_out():
                # memory.push(state, action, None, reward)
                memory.push(state, action, g.get_state(), reward)
            else:
                memory.push(state, action, g.get_state(), reward)

            if g.timed_out():
                break

            # Perform one step of the optimization (on the target network)
            optimize_model(optimizer, policy_net, target_net, memory)

        # Update the target network, copying all weights and biases in DQN
        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    sys.stdout.write("\r|" + "=" * N_games + "| DONE!\n")
    sys.stdout.flush()


if __name__ == "__main__":
    net_name = "mazeRunner.nn"

    train = True
    if train:
        t0 = time.time()
        model = train_model()
        t1 = time.time()
        print("\nTime to train model = %.2f seconds." % (t1 - t0))
        torch.save(model.state_dict(), net_name)
    else:
        model = DQN()
        model.load_state_dict(torch.load(net_name))
        model.eval()

    g = Game()
    g.play(model, slow=True, max_iter=1000)
