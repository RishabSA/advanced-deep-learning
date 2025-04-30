import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

BATCH_SIZE = 128  # how many transitions to sample per training step
GAMMA = 0.999  # discount factor
EPS_START = 0.9  # starting value of epsilon (for ε-greedy)
EPS_END = 0.05  # final value of epsilon
EPS_DECAY = 200  # how quickly epsilon decays
TARGET_UPDATE = 10  # how often to sync target network
LR = 5e-4  # learning rate
NUM_EPISODES = 500  # number of training episodes
MEMORY_CAPACITY = 10000  # replay buffer size
MODEL_PATH = "dqn_cartpole.pth"
SHOULD_TRAIN = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Epsilon‑greedy action selection
steps_done = 0


def select_action(state, policy_net, n_actions):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            # pick action with highest Q-value
            return policy_net(state).argmax(dim=1).view(1, 1)
    else:
        # random action
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # mask for non‑final next states
    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool
    )

    # concatenate all non‑final next states
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a) for the actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # V(s_{t+1}) for non‑final states
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )
    # compute expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss
    loss = F.smooth_l1_loss(state_action_values.view(-1), expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    # gradient clipping
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def main():
    # create environments
    env = gym.make("CartPole-v1")
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # set up networks and memory
    policy_net = DQN(n_states, n_actions).to(device)
    target_net = DQN(n_states, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_CAPACITY)

    # track how long each episode lasted
    episode_durations = []

    if os.path.isfile(MODEL_PATH):
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Loaded trained model from '{MODEL_PATH}'")

    if SHOULD_TRAIN:
        for i_episode in range(1, NUM_EPISODES + 1):
            obs = env.reset()[0] if hasattr(env.reset(), "__len__") else env.reset()
            state = torch.tensor([obs], dtype=torch.float32, device=device)
            total_reward = 0

            for t in range(1, 10000):  # cap steps per episode
                action = select_action(state, policy_net, n_actions)
                # step the environment
                result = env.step(action.item())
                if len(result) == 5:
                    next_obs, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    next_obs, reward, done, _ = result

                total_reward += reward
                reward_t = torch.tensor([reward], device=device)

                if not done:
                    next_state = torch.tensor(
                        [next_obs], dtype=torch.float32, device=device
                    )
                else:
                    next_state = None

                # store in replay buffer
                memory.push(state, action, next_state, reward_t)
                state = next_state

                # perform one step of optimization
                optimize_model(memory, policy_net, target_net, optimizer)

                if done:
                    episode_durations.append(t)
                    break

            # sync target network
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            print(
                f"Episode {i_episode}\tDuration: {episode_durations[-1]}  Total reward: {total_reward:.2f}"
            )

        torch.save(policy_net.state_dict(), MODEL_PATH)

        print("▶️ Training complete")
        env.close()

        plt.figure(figsize=(8, 5))
        plt.plot(episode_durations)
        plt.title("Episode Duration over Time")
        plt.xlabel("Episode")
        plt.ylabel("Duration (timesteps)")
        plt.show()

    # Test the trained agent
    test_env = gym.make("CartPole-v1", render_mode="human")
    for i in range(3):
        obs = test_env.reset()[0]
        state = torch.tensor([obs], dtype=torch.float32, device=device)
        for t in range(1, 1000):
            # pick best action (no exploration)
            action = policy_net(state).argmax(dim=1).view(1, 1)
            result = test_env.step(action.item())
            if len(result) == 5:
                next_obs, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = result

            test_env.render()
            if done:
                break
            state = torch.tensor([next_obs], dtype=torch.float32, device=device)
    test_env.close()


if __name__ == "__main__":
    main()
