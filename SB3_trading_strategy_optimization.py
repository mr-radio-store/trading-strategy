# =====================================================
# Dynamic Trading Strategy Optimization with SB3 (PPO)
# =====================================================

import gym
import numpy as np
from gym import spaces

import matplotlib
matplotlib.use("Agg")  # headless-safe for RPi5
import matplotlib.pyplot as plt

from stable_baselines3 import PPO


# =====================================================
# 1. Dynamic Trading Environment
# =====================================================
class DynamicTradingEnv(gym.Env):
    """
    State-based dynamic trading environment with stochastic prices.
    State vector: [price, moving_avg, position, cash]
    Action: continuous [-1, 1] -> sell (-1), hold (0), buy (+1)
    """
    def __init__(self, initial_cash=10000):
        super().__init__()

        self.initial_cash = initial_cash
        self.max_steps = 200
        self.dt = 1

        # Observation space: price, moving average, position, cash
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, 0]),
            high=np.array([np.inf, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )

        # Action space: continuous sell/buy
        self.action_space = spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.step_count = 0
        self.cash = self.initial_cash
        self.position = 0  # number of shares
        self.total_value = self.cash
        self.history = []

        # Initialize dynamic price
        self.price = 50 + np.random.randn()
        self.prices = [self.price]

        return self._get_obs()

    def step(self, action):
        action = float(np.clip(action[0], -1, 1))

        # Dynamic price update: random walk + noise + occasional spikes
        noise = np.random.randn() * 0.5
        spike = 0
        if np.random.rand() < 0.05:  # 5% chance spike
            spike = np.random.randn() * 3
        self.price += 0.2*np.sin(0.2*self.step_count) + noise + spike
        self.price = max(1.0, self.price)  # price can't go below 1
        self.prices.append(self.price)

        # Convert action to number of shares
        shares = action * 10
        cost = shares * self.price

        # Transaction cost (0.1%)
        cost_with_fee = cost * 1.001

        # Update cash and position
        self.cash -= cost_with_fee
        self.position += shares

        # Total portfolio value
        self.total_value = self.cash + self.position * self.price

        # Reward: change in total value
        if self.step_count == 0:
            reward = 0
        else:
            reward = self.total_value - self.history[-1]

        self.history.append(self.total_value)
        self.step_count += 1
        done = self.step_count >= self.max_steps

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        idx = min(self.step_count, len(self.prices)-1)
        price = self.prices[idx]
        moving_avg = np.mean(self.prices[max(0, idx-5):idx+1])
        return np.array([price, moving_avg, self.position, self.cash], dtype=np.float32)


# =====================================================
# 2. Train PPO
# =====================================================
def train():
    env = DynamicTradingEnv()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        gamma=0.99,
        n_steps=256,
        batch_size=64,
        verbose=1
    )

    model.learn(total_timesteps=80_000)
    model.save("ppo_dynamic_trading")

    return env, model


# =====================================================
# 3. Evaluate & Save Figure
# =====================================================
def evaluate(env, model, filename="dynamic_trading.png"):
    obs = env.reset()
    price_log, value_log, action_log = [], [], []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)

        price_log.append(obs[0])
        value_log.append(env.total_value)
        action_log.append(action[0])

    plt.figure(figsize=(10, 5))
    plt.plot(price_log, label="Stock Price")
    plt.plot(value_log, label="Portfolio Value")
    plt.plot(action_log, label="Action (Buy/Sell)")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.title("Dynamic Trading Strategy Optimization (RL)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"Saved {filename}")


# =====================================================
# 4. Main
# =====================================================
if __name__ == "__main__":
    env, model = train()
    evaluate(env, model)
