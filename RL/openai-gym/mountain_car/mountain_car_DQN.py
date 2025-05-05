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

BATCH_SIZE = 64  # Number of experiences (current state, action, reward, next state) to sample per training step
GAMMA = 0.99  # Discount factor
EPS_START = 0.9  # Epsilon for epsilon-greedy
EPS_END = 0.05
EPS_DECAY = 2000
TARGET_UPDATE = 5  # How often to sync target and policy network parameters
LR = 5e-4
NUM_EPISODES = 1000  # Training episodes
MEMORY_CAPACITY = 10000  # Memory Replay Buffer Capacity
EPISODE_STEP_LIMIT = 200  # After how many steps to end episode early
MODEL_PATH = "dqn_mountain_car.pth"
SHOULD_TRAIN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transition holds one experience that the agent observes at a single timestep
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


# Stores the past transitions observed recently
# By sampling from it randomly, the transitions that build up a batch are decorrelated.
# Greatly stabilizes and improves the DQN training procedure
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        self.buffer.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        # Get batch_size random sample from the replay memory buffer
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        # n_outputs is the number of possible actions the agent can take
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  #  Returns the Q-values for each state, action


# Epsilon‑greedy action selection
steps_done = 0


def select_action(state, policy, n_actions):
    global steps_done

    # Exponentially decay the value of epsilon from start to end
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(
        -1.0 * steps_done / EPS_DECAY
    )
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            # Pick action with highest Q-value from policy
            return policy(state).argmax(dim=1).view(1, 1)
    else:
        # Random action for exploration
        return torch.tensor(
            [[random.randrange(n_actions)]], device=device, dtype=torch.long
        )


def optimize_model(memory, policy, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return  # Wait until the replay memory buffer is filled with at least a batch to start optimizing

    # Sample a batch of transitions from Replay Memory
    transitions = memory.sample(BATCH_SIZE)

    # Transposes the Transition tuples into a single Transition which has tuples of all the individual elements
    # batch.state - tuple of all states
    # batch.action - tuple of all actions
    # batch.reward - tuple of all rewards
    # batch.next_state - tuple of all next states
    batch = Transition(*zip(*transitions))

    # Mask for non‑final next states - Bool tensor indicating which transitions aren’t terminal (where next_state is not None)
    non_final_mask = torch.tensor(
        tuple(s is not None for s in batch.next_state), device=device, dtype=torch.bool
    )

    # Concatenate all non‑final next states: (num_non_final, state_dim)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    # (batch_size, state_dim)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) with the policy network for the actions taken
    # Compute Q-values of every state for all actions (shape: (batch_size, n_actions)), then select the Q-value corresponding to each sample’s taken action, yielding a (batch_size, 1) tensor.
    state_action_values = policy(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for non-final states using the target network
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    # Feeds all non-final next states through the target network and takes the maximum Q-value over actions for each
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach()
    )

    # Compute expected Q values via Bellman equation
    # r + γQ^\pi(s^{\prime},\pi(s^{\prime}))

    # We use the **next state** $s^{\prime}$ because of the Bellman equation: the true value of (s,a) is the immediate reward r plus the discounted value of the *best* next action in $s^{\prime}$.
    # By adding $r$ to $\gamma \max_{a^{\prime}} Q(s^{\prime},a)$, we form a one-step “ground truth” that the policy net should move toward.

    # Discount factor and add observed reward to get "optimal" decision
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss between policy network’s predicted Q-values and target network's Q-values
    loss = F.smooth_l1_loss(state_action_values.view(-1), expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    for param in policy.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()


def main():
    env = gym.make("MountainCar-v0")
    n_states = env.observation_space.shape[0]  # 2 state values: position and velocity
    n_actions = (
        env.action_space.n
    )  # 3: Accelerate left no acceleration, accelerate right

    policy = DQN(n_states, n_actions).to(device)
    target_net = DQN(n_states, n_actions).to(device)
    target_net.load_state_dict(policy.state_dict())
    target_net.eval()

    optimizer = optim.Adam(params=policy.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_CAPACITY)

    # track how long each episode lasted
    episode_durations = []

    if os.path.isfile(MODEL_PATH):
        policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        target_net.load_state_dict(policy.state_dict())
        print(f"Loaded trained model from '{MODEL_PATH}'")

    if SHOULD_TRAIN:
        for i_episode in range(1, NUM_EPISODES + 1):
            obs = env.reset()[0]  # Gymnasium returns (obs, info)
            state = torch.tensor([obs], dtype=torch.float32, device=device)
            total_reward = 0

            for t in range(1, EPISODE_STEP_LIMIT):  # Limit steps per episode
                action = select_action(state, policy, n_actions)
                # step the environment
                result = env.step(action.item())

                observation, reward, terminated, truncated, _ = result
                reward_t = torch.tensor([reward], device=device, dtype=torch.float32)
                total_reward += reward
                done = terminated or truncated

                if done:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)

                # Store the current experience in replay buffer
                memory.push(state, action, next_state, reward_t)
                state = next_state
                optimize_model(memory, policy, target_net, optimizer)

                if done:
                    episode_durations.append(t)
                    break

            # Sync target network parameters with policy network parameters
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy.state_dict())

            print(f"Episode {i_episode} Total reward: {total_reward:.2f}")

        torch.save(policy.state_dict(), MODEL_PATH)

        print("Training complete")
        env.close()

        plt.figure(figsize=(8, 5))
        plt.plot(episode_durations)
        plt.title("Episode Duration over Time")
        plt.xlabel("Episode")
        plt.ylabel("Duration (timesteps)")
        plt.show()

    # Test the trained agent
    test_env = gym.make("MountainCar-v0", render_mode="human")
    for i in range(3):
        obs = test_env.reset()[0]
        state = torch.tensor([obs], dtype=torch.float32, device=device)
        for t in range(1, EPISODE_STEP_LIMIT):
            # pick best action (no exploration)
            action = policy(state).argmax(dim=1).view(1, 1)
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
