import gymnasium as gym
import time

env = gym.make("CartPole-v1", render_mode="human")

episodes = 10

for i in range(episodes):
    # Observation is what the agent should see info is additional data points
    observation, info = env.reset()
    episode_over = False

    while not episode_over:
        # Action space is all of the possible actions that you can take
        # Sample() gets a random action
        # action = env.action_space.sample()

        # 0 - left | 1 - right

        # observation[2] = pole_angle
        action = 1 if observation[2] > 0 else 0

        # Truncated is you survived the entire episode (time limit)
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

        time.sleep(0.01)

env.close()
